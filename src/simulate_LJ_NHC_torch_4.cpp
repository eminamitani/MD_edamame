#include <iostream>
#include <iomanip>

#include <chrono>

#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <vector>

#include "NoseHooverThermostat.hpp"
#include "config.h"
#include "Atoms.hpp"
#include "LJ.hpp"
#include "xyz.hpp"
#include <torch/torch.h>

//---------------------------------------------------------------
constexpr int N = 1000;         // 粒子数
//---------------------------------------------------------------
constexpr auto deg = 3;         // 空間次元
constexpr auto dt  = 5e-3;      // 時間刻み
constexpr auto N_A = N*4/5;     // A粒子の数（80:20のKob-Andersen mixture）
//---------------------------------------------------------------
constexpr auto rho = 1.2;       // 数密度
const double Lbox = std::pow(N/rho, 1.0/deg);
                                // シミュレーションボックスの一辺の大きさ
const double Linv = 1.0/Lbox;   // Lboxの逆数
//---------------------------------------------------------------
enum {X, Y, Z};
//---------------------------------------------------------------
constexpr double margin = 0.7;
constexpr double cutoff = 2.0;
NeighbourList NL(torch::tensor(cutoff), torch::tensor(margin), torch::kCPU);
//---------------------------------------------------------------
//熱浴
double tau = 1.0;
double targ_temp = 2.0;
NoseHooverThermostat thermostats = NoseHooverThermostat(1, targ_temp, tau, torch::kCPU);

Atoms atoms(N, torch::kCPU);
double conf[N][deg], velo[N][deg], force[N][deg];

// 周期境界条件の下で、何個目の箱のミラーに位置しているのか保存する配列
torch::Tensor box_tensor = torch::zeros({N, deg}, torch::kInt64);

void init_atoms(std::mt19937 &mt) {
    // タイプの初期化
    std::vector<std::string> types;
    types.reserve(N);
    for (int i = 0; i < N; ++i) {
        types.push_back((i < N_A) ? "A" : "B");
    }
    std::shuffle(types.begin(), types.end(), mt);
    atoms.set_types(types);

    // 位置の初期化
    const auto ln   = std::ceil(std::pow(N, 1.0/deg));
    const auto haba = Lbox/ln;
    torch::Tensor positions = torch::zeros({N, deg}, torch::kFloat64);

    for (int i=0; i<N; i++) {
        const int iz = std::floor(i/(ln*ln));
        const int iy = std::floor((i - iz*ln*ln)/ln);
        const int ix = i - iz*ln*ln - iy*ln;

        positions[i][X] = haba*0.5 + haba * ix;
        positions[i][Y] = haba*0.5 + haba * iy;
        positions[i][Z] = haba*0.5 + haba * iz;
    }
    atoms.set_positions(positions);
    atoms.set_box_size(torch::tensor(Lbox));
    atoms.apply_pbc();
}

void load() {
    for (int i = 0; i < N; i ++) {
        for (int j = 0; j < deg; j ++) {
            conf[i][j] = atoms.positions()[i][j].item<double>();
            velo[i][j] = atoms.velocities()[i][j].item<double>();
            force[i][j] = atoms.forces()[i][j].item<double>();
        }
    }
}

void set() {
    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    atoms.set_positions(torch::from_blob(conf, {N, deg}, options).clone());
    atoms.set_velocities(torch::from_blob(velo, {N, deg}, options).clone());
    atoms.set_forces(torch::from_blob(force, {N, deg}, options).clone());
}

//---------------------------------------------------------------
inline void remove_drift() {
    double vel1 = 0.0, vel2 = 0.0, vel3 = 0.0;

    // 系全体の速度を計算する
    for (int i=0; i<N; i++) {
        vel1 += velo[i][X];
        vel2 += velo[i][Y];
        vel3 += velo[i][Z];
    }
    vel1 /= N;
    vel2 /= N;
    vel3 /= N;
    // 各粒子の速度から、系全体の速度/Nを引いておく
    for (int i=0; i<N; i++) {
        velo[i][X] -= vel1;
        velo[i][Y] -= vel2;
        velo[i][Z] -= vel3;
    }
}

void init_vel_MB(const double T_targ, std::mt19937 &mt) {
    load();

    std::normal_distribution<double> dist_trans(0.0, std::sqrt(T_targ));
    for (int i=0; i<N; i++) {
        velo[i][X] = dist_trans(mt);
        velo[i][Y] = dist_trans(mt);
        velo[i][Z] = dist_trans(mt);
    }
    remove_drift();

    set();
}
//---------------------------------------------------------------
// LJパラメータを保存しておくルックアップ・テーブル
constexpr double MBLJ_sij1[2][2] = {
    {1.0, 0.8},
    {0.8, 0.88}
};
constexpr double MBLJ_energy[2][2] = {
    {1.0, 1.5},
    {1.5, 0.5}
};
//---------------------------------------------------------------
// LJポテンシャルとその一階微分
double LJpotential(const double rij1, const double sij1) {
    const double rij2 = rij1 * rij1;
    const double rij6 = rij2 * rij2 * rij2;
    const double sij2 = sij1 * sij1;
    const double sij6 = sij2 * sij2 * sij2;
    return 4.0 * sij6 * (sij6 - rij6)/(rij6 * rij6);
}
double deriv_1st_LJpotential(const double rij1, const double sij1) {
    const double rij2 = rij1 * rij1;
    const double rij6 = rij2 * rij2 * rij2;
    const double sij2 = sij1 * sij1;
    const double sij6 = sij2 * sij2 * sij2;
    return -24.0/rij1 * sij6 * (2.0 * sij6 - rij6)/(rij6 * rij6);
}
//---------------------------------------------------------------
void calc_force() {
    // 力を計算する前に force をゼロ埋め
    atoms.set_forces(torch::zeros({N, 3}, torch::kFloat64));
    auto types = atoms.types(); 

    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < deg; ++d) {
            force[i][d] = 0.0;
        }
    }

    // 力の計算のループ
    for (int i=0; i<N; i++) {
        // 粒子 i の粒子種を判定
        const int si = (types[i] == "A") ? 0 : 1;
        torch::Tensor mask = (NL.source_index() == i);
        torch::Tensor target_indices = NL.target_index().index({mask});

        for (int64_t p = 0; p < target_indices.size(0); p ++) {
            // 粒子jのインデックスを取得
            const int j = target_indices[p].item<int>();
            if (j < i) continue;

            // 粒子 j の粒子種を判定
            const int sj = (types[j] == "A") ? 0 : 1; 

            // ij 間の距離の2乗を計算
            double dx = conf[i][X] - conf[j][X];
            double dy = conf[i][Y] - conf[j][Y];
            double dz = conf[i][Z] - conf[j][Z];
            dx -= Lbox * std::floor(dx * Linv + 0.5);
            dy -= Lbox * std::floor(dy * Linv + 0.5);
            dz -= Lbox * std::floor(dz * Linv + 0.5);
            const double rij2 = dx*dx + dy*dy + dz*dz;

            // カットオフ距離を計算
            // 粒子iと粒子jが同種粒子なら1.5
            // 粒子iと粒子jが異種粒子なら2.0
            const double rc1 = (si==sj ? 1.5 : 2.0);

            // 粒子iと粒子jが相互作用するか判定
            if (rij2 < rc1 * rc1) {
                const double rij1 = std::sqrt(rij2);
                const double sij1 = MBLJ_sij1[si][sj];

                double deriv_1st = deriv_1st_LJpotential(rij1, sij1) - deriv_1st_LJpotential(rc1, sij1);
                deriv_1st *= MBLJ_energy[si][sj];

                force[i][X] -= deriv_1st * dx / rij1;
                force[i][Y] -= deriv_1st * dy / rij1;
                force[i][Z] -= deriv_1st * dz / rij1;
                force[j][X] += deriv_1st * dx / rij1;
                force[j][Y] += deriv_1st * dy / rij1;
                force[j][Z] += deriv_1st * dz / rij1;
            }
        }
    }
}
//---------------------------------------------------------------
double calc_potential() {
    auto types = atoms.types(); 
    double ans = 0.0;
    for (int i=0; i<N-1; i++) {
        const int si = (types[i] == "A") ? 0 : 1;
        torch::Tensor mask = (NL.source_index() == i);
        torch::Tensor target_indices = NL.target_index().index({mask});

        for (int64_t p = 0; p < target_indices.size(0); p ++) {
            const int j = target_indices[p].item<int>();
            
            if (j < i) continue;

            const int sj = (types[j] == "A") ? 0 : 1; 

            double dx = conf[i][X] - conf[j][X];
            double dy = conf[i][Y] - conf[j][Y];
            double dz = conf[i][Z] - conf[j][Z];
            dx -= Lbox * std::floor(dx * Linv + 0.5);
            dy -= Lbox * std::floor(dy * Linv + 0.5);
            dz -= Lbox * std::floor(dz * Linv + 0.5);

            const double rij2 = dx*dx + dy*dy + dz*dz;
            const double rc1 = (si==sj ? 1.5 : 2.0);
            if (rij2 < rc1 * rc1) {
                const double rij1 = std::sqrt(rij2);
                const double sij1 = MBLJ_sij1[si][sj];
                double potential = LJpotential(rij1, sij1) - LJpotential(rc1, sij1) - deriv_1st_LJpotential(rc1, sij1) * (rij1 - rc1);
                ans += potential * MBLJ_energy[si][sj];
            }
        }
    }
    return ans;
}
//---------------------------------------------------------------
void print_energies(const long t, bool is_save) {
    auto box = box_tensor.accessor<long, 2>();
    // ポテンシャルエネルギーを計算
    const double U = calc_potential();

    // 運動エネルギーを計算
    double K = 0.0;
    for (int i=0; i<N; i++) {
        K += 0.5 * (velo[i][X]*velo[i][X]
                    + velo[i][Y]*velo[i][Y]
                    + velo[i][Z]*velo[i][Z]);
    }

    // 時刻、1粒子当たりの運動エネルギー、1粒子当たりのポテンシャルエネルギー、1粒子当たりの全エネルギーを出力
    std::cout << std::setprecision(15) << std::scientific
              << dt * t << ","
              << K/N << ","
              << U/N << ","
              << (K + U)/N << std::endl;

    if (is_save) {
        //構造の保存
        // ファイル名をステップ数で生成
        std::string filename = "LJ_NHC_traj.xyz";
        std::ofstream ofs(filename, std::ios_base::app);
        if (!ofs) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return;
        }

        // xyz形式のヘッダーを書き込む
        ofs << N << std::endl;
        ofs << "Lattice=\"" << Lbox << " 0.0 0.0 0.0 " << Lbox << " 0.0 0.0 0.0 " << Lbox << "\" "
            << "Properties=species:S:1:pos:R:3 Time=" << dt * t << std::endl;

        // 各原子の情報を書き込む
        auto types = atoms.types();
        for (int i = 0; i < N; ++i) {
            // 粒子種を判定 (i < N_A なら 'A', それ以外は 'B')
            const char species = types[i][0];

            // 周期境界条件を展開した座標を計算
            double unfolded_x = conf[i][X] + box[i][X] * Lbox;
            double unfolded_y = conf[i][Y] + box[i][Y] * Lbox;
            double unfolded_z = conf[i][Z] + box[i][Z] * Lbox;

            // ファイルに書き込む
            ofs << species << " "
                << unfolded_x << " "
                << unfolded_y << " "
                << unfolded_z << std::endl;
        }
        ofs.close();
    }
}

//---------------------------------------------------------------
void NVT(const double tsim, bool is_save) {
    load();

    calc_force();

    set();

    thermostats.setup(torch::tensor(3 * N - 3));

    const auto logbin = std::pow(10.0, 1.0/9);
    int counter = 5;
    auto checker = 1e-3 * std::pow(logbin, counter);

    long t = 0;
    load();
    print_energies(t, is_save);
    set();

    const long steps = tsim/dt;

    torch::Tensor dt_tensor = torch::tensor(dt);

    while (t < steps) {
        thermostats.update(atoms, dt_tensor);
        atoms.velocities_update(dt_tensor);
        atoms.positions_update(dt_tensor, box_tensor);

        load();

        // 更新した位置においてマージンを飛び出した粒子がいないか判定
        NL.update(atoms);

        calc_force();

        set();

        atoms.velocities_update(dt_tensor);
        thermostats.update(atoms, dt_tensor);

        load();

        t++;

        if (dt*t > checker) {
            checker *= logbin;
            print_energies(t, is_save);
        }
    }
}
//---------------------------------------------------------------
int main() {
    double t_eq = 6.68 * 50;

    // 疑似乱数生成器を適当に初期化
    std::mt19937 mt(123456789);

    // 正方格子に粒子を配置する
    init_atoms(mt);

    load();

    // 初期化した粒子位置で隣接リストを構築する
    NL.generate(atoms);

    // 初期速度は、温度T = 2.0のマクスウェル・ボルツマン分布から引っ張ってくる
    init_vel_MB(2.0, mt);

    // t = 1e5の長さのNVEシミュレーションを実行（時間を測りながら）
    auto start = std::chrono::system_clock::now();
    NVT(t_eq, false);  //equilibration run
    std::cout << "equilibration run 終了" << std::endl;
    NVT(t_eq, true);  //production run
    std::cout << "production run 終了" << std::endl;
    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    // 実行時間を出力
    std::cout << elapsed << std::endl;
}
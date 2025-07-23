#include <iostream>
#include <iomanip>

#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <vector>

#include "xyz.hpp"
#include "config.h"
#include "Atoms.hpp"

#include <torch/torch.h>

//---------------------------------------------------------------
constexpr auto N = 1000;         // 粒子数
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
double conf[N][deg], velo[N][deg], force[N][deg];
                                // 粒子の位置、粒子の速度、粒子にかかる力を保持する配列
enum {X, Y, Z};
//---------------------------------------------------------------
void init_lattice() {
    const auto ln   = std::ceil(std::pow(N, 1.0/deg));
    const auto haba = Lbox/ln;

    for (int i=0; i<N; i++) {
        const int iz = std::floor(i/(ln*ln));
        const int iy = std::floor((i - iz*ln*ln)/ln);
        const int ix = i - iz*ln*ln - iy*ln;

        // 正方格子に粒子を配置
        conf[i][X] = haba*0.5 + haba * ix;
        conf[i][Y] = haba*0.5 + haba * iy;
        conf[i][Z] = haba*0.5 + haba * iz;

        // minimum-image convention
        for (int d=0; d<deg; d++) {
            conf[i][d] -= Lbox * std::floor(conf[i][d] * Linv + 0.5);
        }
    }
}
void init_species(std::mt19937 &mt) {
    // 0,1,...,N-1が格納されたstd::vector配列を初期化
    std::vector<int> v(N);
    std::iota(v.begin(), v.end(), 0);

    // その配列をシャッフル
    std::shuffle(v.begin(), v.end(), mt);

    for (int i=0; i<N; i+=2) {
        // シャッフルされた配列内で隣接する粒子インデックスペアを使って
        // 粒子位置を交換する
        const int id0 = v[i];
        const int id1 = v[i+1];

        // 交換前の位置を一時的に保存しておく        
        const double position0_X = conf[id0][X];
        const double position0_Y = conf[id0][Y];
        const double position0_Z = conf[id0][Z];

        const double position1_X = conf[id1][X];
        const double position1_Y = conf[id1][Y];
        const double position1_Z = conf[id1][Z];

        // 交換する
        conf[id0][X] = position1_X;
        conf[id0][Y] = position1_Y;
        conf[id0][Z] = position1_Z;
        conf[id1][X] = position0_X;
        conf[id1][Y] = position0_Y;
        conf[id1][Z] = position0_Z;
    }
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
    std::normal_distribution<double> dist_trans(0.0, std::sqrt(T_targ));
    for (int i=0; i<N; i++) {
        velo[i][X] = dist_trans(mt);
        velo[i][Y] = dist_trans(mt);
        velo[i][Z] = dist_trans(mt);
    }
    remove_drift();
}
//---------------------------------------------------------------
int main() {
    // 疑似乱数生成器を適当に初期化
    std::mt19937 mt(123456789);

    // 正方格子に粒子を配置する
    init_lattice();

    // A粒子とB粒子の初期位置はランダムに混ぜておく
    init_species(mt);

    //位置、速度、力をテンソルに変換
    torch::Tensor position_tensor = torch::from_blob(conf, {N, deg}, kRealType);
    torch::Tensor velocity_tensor = torch::from_blob(velo, {N, deg}, kRealType);
    torch::Tensor force_tensor = torch::from_blob(force, {N, deg}, kRealType);

    std::vector<std::string> types;
    types.reserve(N);
    for(int i = 0; i < N; i ++) {
        if (i < N_A) {
            types.push_back("A");
        }
        else {
            types.push_back("B");
        }
    }

    Atoms atoms = Atoms(N, torch::kCPU);
    atoms.set_box_size(torch::tensor(Lbox, kRealType));
    atoms.set_types(types);
    atoms.set_positions(position_tensor);
    atoms.set_velocities(velocity_tensor);
    atoms.set_forces(force_tensor);
    atoms.set_potential_energy(torch::tensor(0.0, kRealType));

    std::string save_path = "../data/LJ_random_structure.xyz";

    xyz::save_atoms(save_path, atoms);
}
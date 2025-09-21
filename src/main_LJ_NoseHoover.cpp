#include "MD.hpp"
#include "Atoms.hpp"
#include "NoseHooverThermostat.hpp"
#include "config.h"
#include <chrono>
#include <thread>

int main(){
    //デバイス
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    //系の設定
    const IntType num_atoms = 1000;         //粒子数
    const RealType rho = 1.2;               //数密度
    const RealType ratio = 0.8;             //粒子Aの割合

    //シミュレーション定数
    const RealType dt = 5e-3;               //タイムステップ
    const RealType cutoff = 2.0;            //カットオフ距離
    const RealType margin = 0.7;            //隣接リストのマージン

    const RealType tau = 1.0;
    const IntType chain_length = 1;

    const RealType T_0 = 2.0;                 
    const RealType T_targ = 2.0;            //ターゲット温度

    const RealType tau_alpha = 6.68;
    const RealType t_eq = tau_alpha * 50;
    const RealType t_pr = t_eq;

    //現在時刻を記録
    auto start = std::chrono::steady_clock::now();

    //熱浴の初期化
    NoseHooverThermostat thermostat(chain_length, T_0, tau, device);

    //LJユニットの作成
    Atoms LJ_unit = Atoms::make_LJ_unit(num_atoms, ratio, rho, device);

    //MDオブジェクトの実体化
    MD md = MD(dt, cutoff, margin, LJ_unit, device);

    //T = T_0で緩和
    thermostat.set_temp(T_0);
    md.NVT_LJ(t_eq, thermostat, "log", false, "./LJ_2_eq1.xyz");
    md.reset_step();

    //T = T_targでproduction run
    thermostat.set_temp(T_targ);
    md.save_unwrapped_atoms("./outputs/output_traj.xyz");   //最初の構造を保存
    md.NVT_LJ(t_pr, thermostat, "log", true, "./output_LJ_2.xyz");

    //終了時刻を記録
    auto end = std::chrono::steady_clock::now();

    //実行時間を計算
    double elapsed_s = std::chrono::duration<double>(end - start).count();

    std::cout << "処理にかかった時間：" << elapsed_s << "s" << std::endl;
}
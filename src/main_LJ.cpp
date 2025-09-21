#include "MD.hpp"
#include "Atoms.hpp"
#include "BussiThermostat.hpp"
#include "config.h"
#include <chrono>
#include <thread>

int main(){
    //デバイス
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    //系の設定
    const IntType num_atoms = 500;          //粒子数
    const RealType rho = 1.2;               //数密度
    const RealType ratio = 0.8;             //粒子Aの割合

    //シミュレーション定数
    const RealType dt = 1e-3;               //タイムステップ
    const RealType cutoff = 2.0;            //カットオフ距離
    const RealType margin = 1.0;            //隣接リストのマージン

    const RealType tau = 0.5;              //Bussi熱浴の時定数

    const RealType T_0 = 1.0;
    const RealType T_1 = 2.0;                   
    const RealType T_targ = 2.0;            //ターゲット温度

    const RealType t_eq = 1.36e+4;          //緩和時間
    const RealType t_pr = t_eq;

    //現在時刻を記録
    auto start = std::chrono::steady_clock::now();

    //熱浴の初期化
    BussiThermostat bussi_thermostat(T_0, tau, device);

    //LJユニットの作成
    Atoms LJ_unit = Atoms::make_LJ_unit(num_atoms, ratio, rho, device);

    //MDオブジェクトの実体化
    MD md = MD(dt, cutoff, margin, LJ_unit, device);

    //T = T_0で緩和
    bussi_thermostat.set_temp(T_0);
    md.NVT_LJ(t_eq, bussi_thermostat, "log", false, "./LJ_2_eq1.xyz");
    md.reset_step();

    //T = T_1で緩和
    bussi_thermostat.set_temp(T_1);
    md.NVT_LJ(t_eq, bussi_thermostat, "log", false, "./LJ_2_eq2.xyz");
    md.reset_step();

    //T = T_targでproduction run
    bussi_thermostat.set_temp(T_targ);
    md.save_unwrapped_atoms("./outputs/output_traj.xyz");   //最初の構造を保存
    md.NVT_LJ(t_pr, bussi_thermostat, "log", true, "./output_LJ_2.xyz");

    //終了時刻を記録
    auto end = std::chrono::steady_clock::now();

    //実行時間を計算
    double elapsed_s = std::chrono::duration<double>(end - start).count();

    std::cout << "処理にかかった時間：" << elapsed_s << "s" << std::endl;
}
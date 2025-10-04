#include "MD.hpp"
#include "BussiThermostat.hpp"
#include "config.h"
#include <chrono>
#include <thread>

int main(){
    //デバイス
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    //定数
    const RealType dt = 0.5;                //タイムステップ (fs)
    const RealType cutoff = 5.0;            //カットオフ距離 (Å)
    const RealType margin = 1.0;            //隣接リストのマージン (Å)

    const RealType tau = 1e+2;              //Bussi熱浴の時定数

    const RealType T_0 = 3300.0;            //初期温度 (K)
    const RealType T_targ = 1000.0;         //目標温度 (K)
    const RealType cooling_rate = 1.0;      //冷却速度 (K/fs)
    const RealType t_eq = 2e+4;             //緩和時間 (fs)
    const RealType t_sim = 5e+5;            //production runの時間 (fs)

    //パス
    const std::string data_path = "./output_NS2_eq2.xyz";                  //初期構造のパス
    const std::string model_path = "./models/deployed_model_Na2O-SiO2.pt";  //NNPモデルのパス

    //保存先
    const std::string traj_path = "./NS2_T1000.xyz";

    //現在時刻を記録
    auto start = std::chrono::steady_clock::now();

    //熱浴の初期化
    BussiThermostat bussi_thermostat(T_0, tau, device);

    //MDオブジェクトの実体化
    MD md = MD(dt, cutoff, margin, data_path, model_path, device);

    //trajectoryファイルの出力先を変更
    md.set_traj_path(traj_path);

    //シミュレーションの実行（冷却）
    md.NVT_anneal(cooling_rate, bussi_thermostat, T_targ, "log", false);
    std::cout << "冷却完了" << std::endl;
    md.reset_step();

    //シミュレーションの実行（緩和）
    md.NVT(t_eq, bussi_thermostat, "log", false);
    std::cout << "緩和完了" << std::endl;
    md.reset_step();

    //シミュレーションの実行（production run）
    md.NVT(t_sim, bussi_thermostat, "log", true);
    md.save_atoms("./NS2_T3000_1.xyz");

    //終了時刻を記録
    auto end = std::chrono::steady_clock::now();

    //実行時間を計算
    double elapsed_s = std::chrono::duration<double>(end - start).count();

    std::cout << "処理にかかった時間：" << elapsed_s << "s" << std::endl;
}
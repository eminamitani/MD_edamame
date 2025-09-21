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
    const RealType T_targ = 300.0;          //目標温度 (K)
    const RealType cooling_rate = 2e-3;     //冷却速度 (K / fs)
    const RealType t_eq = 2e+5;             //緩和時間 (fs)

    //パス
    const std::string data_path = "./data/sample_NS4.xyz";                  //初期構造のパス
    const std::string model_path = "./models/deployed_model_Na2O-SiO2.pt";  //NNPモデルのパス

    //現在時刻を記録
    auto start = std::chrono::steady_clock::now();

    //熱浴の初期化
    BussiThermostat bussi_thermostat(T_0, tau, device);

    //MDオブジェクトの実体化
    MD md = MD(dt, cutoff, margin, data_path, model_path, device);

    //シミュレーションの実行（緩和）
    md.NVT(t_eq, bussi_thermostat, 100, false, "./output_NS4_eq1.xyz");
    std::cout << "緩和完了" << std::endl;

    //シミュレーションの実行（冷却）
    md.NVT_anneal(cooling_rate, bussi_thermostat, T_targ, 100, false, "./output_NS4_quenched.xyz");
    std::cout << "冷却完了" << std::endl;

    //シミュレーションの実行（緩和）
    md.NVT(t_eq, bussi_thermostat, 100, false, "./output_NS4_eq2.xyz");
    std::cout << "緩和完了" << std::endl;

    //終了時刻を記録
    auto end = std::chrono::steady_clock::now();

    //実行時間を計算
    double elapsed_s = std::chrono::duration<double>(end - start).count();

    std::cout << "処理にかかった時間：" << elapsed_s << "s" << std::endl;
}
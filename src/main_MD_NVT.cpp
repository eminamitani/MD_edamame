#include "MD.hpp"
#include "BussiThermostat.hpp"
#include "config.h"
#include <chrono>
#include <thread>

int main(){
    //デバイス
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    //定数
    const RealType dt = 0.5;
    const RealType cutoff = 5.0;
    const RealType margin = 1.0;

    const RealType tau = 1e+2;

    const RealType T_0 = 3300.0;            //初期温度
    const RealType T_targ = 300.0;          //目標温度
    const RealType cooling_rate = 2e-3;     //冷却速度 (K / fs)
    const RealType t_eq = 2e+5;             //緩和時間 (fs)

    //パス
    const std::string data_path = "./data/NS22_5.xyz";
    const std::string model_path = "./models/deployed_model_Na2O-SiO2.pt";

    //現在時刻を記録
    auto start = std::chrono::steady_clock::now();

    //熱浴の初期化
    BussiThermostat bussi_thermostat(T_0, tau, device);

    //MDオブジェクトの実体化
    MD md = MD(dt, cutoff, margin, data_path, model_path, device);

    //速度の初期化
    md.set_temp(T_0);

    //シミュレーションの実行
    md.MQ_log(t_eq, cooling_rate, bussi_thermostat, T_targ);

    //終了時刻を記録
    auto end = std::chrono::steady_clock::now();

    //実行時間を計算
    double elapsed_s = std::chrono::duration<double>(end - start).count();

    std::cout << "処理にかかった時間：" << elapsed_s << "s" << std::endl;
}
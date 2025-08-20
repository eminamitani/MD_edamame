#include "MD.hpp"
#include "NoseHooverThermostat.hpp"
#include "config.h"
#include <chrono>
#include <thread>

int main(){
    //デバイス
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    //定数
    const RealType dt = 0.1;
    const RealType cutoff = 5.0;
    const RealType margin = 1.0;

    const RealType tau = dt * 50;
    const IntType chain_length = 2;
    const RealType temperature = 300.0;

    //パス
    const std::string data_path = "../data/diamond_structure_2.xyz";
    const std::string model_path = "../models/deployed_model.pt";

    //現在時刻を記録
    auto start = std::chrono::steady_clock::now();

    //熱浴の初期化
    NoseHooverThermostat nose_hoover_thermostat(chain_length, temperature, tau, device);

    //MDオブジェクトの実体化
    MD md = MD(dt, cutoff, margin, data_path, model_path, device);

    //速度の初期化
    md.init_vel_MB(300.0);

    //シミュレーションの開始
    md.NVT(1e+5, nose_hoover_thermostat);

    //終了時刻を記録
    auto end = std::chrono::steady_clock::now();

    //実行時間を計算
    double elapsed_s = std::chrono::duration<double>(end - start).count();

    std::cout << "処理にかかった時間：" << elapsed_s << "s" << std::endl;
}
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
    const RealType dt = 1e-3;                //タイムステップ
    const RealType cutoff = 2.0;            //カットオフ距離
    const RealType margin = 0.7;            //隣接リストのマージン

    //現在時刻を記録
    auto start = std::chrono::steady_clock::now();

    //LJユニットの作成
    Atoms LJ_unit = Atoms::make_LJ_unit(num_atoms, ratio, rho, device);

    //MDオブジェクトの実体化
    MD md = MD(dt, cutoff, margin, LJ_unit, device);

    md.NVE_LJ(1e+5, 1.0, 100, false, "./LJ_NVE.xyz");

    //終了時刻を記録
    auto end = std::chrono::steady_clock::now();

    //実行時間を計算
    double elapsed_s = std::chrono::duration<double>(end - start).count();

    std::cout << "処理にかかった時間：" << elapsed_s << "s" << std::endl;
}
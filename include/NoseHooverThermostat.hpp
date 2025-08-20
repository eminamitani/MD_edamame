#ifndef NOSE_HOOVER_THERMOSTAT_HPP
#define NOSE_HOOVER_THERMOSTAT_HPP

#include <functional>
#include <vector>
#include <array>

#include <torch/torch.h>

#include "Atoms.hpp"
#include "config.h"

class NoseHooverThermostat {
    public:
        //コンストラクタ
        NoseHooverThermostat(const IntType length, const torch::Tensor target_tmp, const torch::Tensor tau, torch::Device device = torch::kCPU);
        NoseHooverThermostat(const IntType length, const RealType target_tmp, const RealType tau, torch::Device device = torch::kCPU);
        NoseHooverThermostat();

        //ゲッター
        const IntType length() const { return length_; }
        const torch::Tensor& positions() const { return positions_; }
        const torch::Tensor& masses() const { return masses_; }
        const torch::Tensor& velocities() const { return velocities_; }
        const torch::Tensor& dof() const { return dof_; }
        const torch::Tensor& target_tmp() const { return target_tmp_; }

        //セッター
        void set_target_tmp(const torch::Tensor& temp) { target_tmp_ = temp; }
        void set_target_tmp(const RealType& temp) { target_tmp_ = torch::tensor(temp, device_); }

        //初期化
        void setup(Atoms& atoms);
        void setup(torch::Tensor dof);

        //更新
        void update(Atoms& atoms, const torch::Tensor& dt);
        void update(torch::Tensor& atoms_velocities, const torch::Tensor& kinetic_energy, const torch::Tensor& dt);    //速度のみを使う場合

        NoseHooverThermostat(const NoseHooverThermostat&) = delete;
        NoseHooverThermostat& operator=(const NoseHooverThermostat&) = delete;
        
    private:
        //更新用
        std::function<void(torch::Tensor&, const torch::Tensor&, const torch::Tensor&)> update_function;                  //関数を代入するメンバ変数
        void NHC1(torch::Tensor& atoms_velocities, const torch::Tensor& kinetic_energy, const torch::Tensor& dt);         //能勢フーバーチェイン1
        void NHC2(torch::Tensor& atoms_velocities, const torch::Tensor& kinetic_energy, const torch::Tensor& dt);         //能勢フーバーチェイン2
        void NHCM(torch::Tensor& atoms_velocities, const torch::Tensor& kinetic_energy, const torch::Tensor& dt);

        //変数
        IntType length_;                //チェインの長さM (1, )
        torch::Tensor positions_;       //変位 (M, )
        torch::Tensor masses_;          //質量 (M, )
        torch::Tensor velocities_;      //速度 (M, )
        torch::Tensor forces_;          //力 (M, )
        torch::Tensor tau_;             //緩和時間 (1, )
        torch::Tensor dof_;             //系の自由度 (1, )

        torch::Tensor target_tmp_;      //目標温度 (1, )

        torch::Tensor boltzmann_constant_;  //ボルツマン定数 (1, )
        torch::Device device_;              //計算デバイス
};

#endif
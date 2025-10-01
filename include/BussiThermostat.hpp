/**
* @file BussiThermostat.hpp
* @brief BussiThermostatクラス
*/

#ifndef BUSSI_THERMOSTAT_HPP
#define BUSSI_THERMOSTAT_HPP

#include "Atoms.hpp"
#include "Thermostat.hpp"
#include "config.h"

#include <functional>
#include <random>

#include <torch/torch.h>

class BussiThermostat {
    public:
        BussiThermostat(const torch::Tensor& targ_temp, const torch::Tensor& tau, const torch::Device& device = torch::kCPU);
        BussiThermostat(const RealType& targ_temp, const RealType& tau, const torch::Device& device = torch::kCPU);

        /**
         * @brief 熱浴の温度を取得
         * @return 温度
         * @note 戻り値は0次元のtorch::Tensorです。
         */
        const torch::Tensor& temp() const { return targ_temp_; }
        /**
         * @brief 熱浴の温度を取得
         * @return 温度
         * @note 戻り値はRealType型です。
         */
        RealType temp_real() const { return targ_temp_.item<RealType>(); }

        /**
         * @brief 熱浴のセットアップ
         * 
         * 自由度を初期化します。
         * 使用前に必ず呼んでください。
         * 
         * @param[in] atoms 温度制御する系
         */
        void setup(const Atoms& atoms);
        /**
         * @brief 熱浴のセットアップ
         * 
         * 自由度を初期化します。
         * 使用前に必ず呼んでください。
         * 自由度を直接初期化する時に使用します。
         * 
         * @param[in] dof 自由度
         */
        void setup(const torch::Tensor& dof);

        /**
         * @brief 熱浴の更新
         * @param[out] atoms 温度制御する系
         * @param[in] dt 時間刻み幅
         */
        void update(Atoms& atoms, const torch::Tensor& dt);
        /**
         * @brief 熱浴の更新
         * 
         * Atomsクラスを使わず、速度と運動エネルギーを直接指定する時に指定します。
         * 
         * @param[out] atoms_velocities 温度制御する系の速度 
         * (N, 3)のtorch::Tensor
         * 
         * @param[in] kinetic_energy 温度制御する系の運動エネルギー 
         * 0次元のtorch::Tensor
         * 
         * @param[in] dt 時間刻み幅
         */
        void update(torch::Tensor& atoms_velocities, const torch::Tensor& kinetic_energy, const torch::Tensor& dt);

        /**
         * @brief 目標温度を指定
         * @param[in] targ_temp 目標温度 
         */
        void set_temp(const torch::Tensor& targ_temp) { targ_temp_ = targ_temp; }
        /**
         * @brief 目標温度を指定
         * @param[in] targ_temp 目標温度 
         */
        void set_temp(const RealType& targ_temp);

    private:
        torch::Tensor dof_;
        torch::Tensor tau_;
        torch::Tensor targ_temp_;

        torch::Tensor boltzmann_constant_;

        torch::Device device_;
};

#endif
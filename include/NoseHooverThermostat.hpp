/**
* @file NoseHooverThermostat.hpp
* @brief NoseHooverThermostatクラス
*/

#ifndef NOSE_HOOVER_THERMOSTAT_HPP
#define NOSE_HOOVER_THERMOSTAT_HPP

#include <functional>
#include <vector>
#include <array>

#include <torch/torch.h>

#include "Atoms.hpp"
#include "Thermostat.hpp"
#include "config.h"

class NoseHooverThermostat {
    public:
        //コンストラクタ
        NoseHooverThermostat(const IntType length, const torch::Tensor target_tmp, const torch::Tensor tau, torch::Device device = torch::kCPU);
        NoseHooverThermostat(const IntType length, const RealType target_tmp, const RealType tau, torch::Device device = torch::kCPU);
        NoseHooverThermostat();

        //ゲッター
        /**
         * @brief チェインの長さを取得します
         * @return チェインの長さ
         */
        const IntType length() const { return length_; }
        /**
         * @brief 熱浴の変位を取得します。
         * @return 熱浴の変位
         * @note 戻り値は(M, )のtorch::Tensor
         */
        const torch::Tensor& positions() const { return positions_; }
        /**
         * @brief 熱浴の質量を取得します。
         * @return 熱浴の質量
         * @note 戻り値は(M, )のtorch::Tensor
         */
        const torch::Tensor& masses() const { return masses_; }
        /**
         * @brief 熱浴の速度を取得します。
         * @return 熱浴の速度
         * @note 戻り値は(M, )のtorch::Tensor
         */
        const torch::Tensor& velocities() const { return velocities_; }
        /**
         * @brief 熱浴の自由度を取得します。
         * @return 熱浴の自由度
         * @note 戻り値は0次元のtorch::Tensor
         */
        const torch::Tensor& dof() const { return dof_; }
        /**
         * @brief 目標温度を取得します。
         * @return 温度
         * @note 戻り値は0次元のtorch::Tensor
         */
        const torch::Tensor& temp() const { return target_tmp_; }

        //セッター
        /**
         * @brief 目標温度を設定します。
         * @param[in] target_temp 目標温度
         * 目標温度は0次元のtorch::Tensor
         */
        void set_temp(const torch::Tensor& temp) { target_tmp_ = temp; }
        /**
         * @brief 目標温度を設定します。
         * @param[in] target_temp 目標温度
         */
        void set_temp(const RealType& temp) { target_tmp_ = torch::tensor(temp, device_); }

        //初期化
        /**
         * @brief 熱浴の初期化をします。
         * 
         * 使用前に必ず呼んでください。
         * 自由度を初期化します。
         * 
         * @param[in] atoms 温度制御する系
         */
        void setup(Atoms& atoms);
        /**
         * @brief 熱浴の初期化をします。
         * 
         * 使用前に必ず呼んでください。
         * 自由度を初期化します。
         * 直接自由度を指定する時に使用します。
         * 
         * @param[in] dof 自由度
         */
        void setup(torch::Tensor dof);

        //更新
        /**
         * @brief 熱浴の更新
         * @param[in] atoms 温度制御する系
         * @param[in] dt 時間刻み幅
         */
        void update(Atoms& atoms, const torch::Tensor& dt);
        /**
         * @brief 熱浴の更新
         * 
         * atomsクラスを用いず、温度をそのまま入力する際に使用します。
         * 
         * @param[in] atoms_velocities 制御する速度
         * @param[in] kinetic_energy 運動エネルギー
         * @param[in] dt 時間刻み幅
         */
        void update(torch::Tensor& atoms_velocities, const torch::Tensor& kinetic_energy, const torch::Tensor& dt);    //速度のみを使う場合

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
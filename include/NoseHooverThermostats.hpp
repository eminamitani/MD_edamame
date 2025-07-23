#ifndef NOSE_HOOVER_THERMOSTATS_HPP
#define NOSE_HOOVER_THERMOSTATS_HPP

#include <functional>
#include <vector>
#include <array>

#include <torch/torch.h>

#include "Atoms.hpp"
#include "config.h"

class NoseHooverThermostats {
    public:
        //コンストラクタ
        NoseHooverThermostats(const IntType length, const torch::Tensor target_tmp, const torch::Tensor tau, torch::Device device = torch::kCPU);
        NoseHooverThermostats();

        //ゲッター
        const IntType length() const { return length_; }
        const torch::Tensor& positions() const { return positions_; }
        const torch::Tensor& masses() const { return masses_; }
        const torch::Tensor& velocities() const { return velocities_; }
        const torch::Tensor& dof() const { return dof_; }
        const torch::Tensor& target_tmp() const { return target_tmp_; }

        //初期化
        void setup(Atoms& atoms);
        void setup(torch::Tensor dof);

        //更新
        void update(Atoms& atoms, const torch::Tensor& dt);
        void update(torch::Tensor& atoms_velocities, const torch::Tensor& kinetic_energy, const torch::Tensor& dt);    //速度のみを使う場合

        NoseHooverThermostats(const NoseHooverThermostats&) = delete;
        NoseHooverThermostats& operator=(const NoseHooverThermostats&) = delete;
        
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
        torch::Tensor tau_;
        torch::Tensor dof_;

        torch::Tensor target_tmp_;      //目標温度 (1, )

        torch::Tensor boltzmann_constant_;  //ボルツマン定数 (1, )
        torch::Device device_;              //計算デバイス
};

#endif
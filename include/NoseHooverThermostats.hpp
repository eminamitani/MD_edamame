#ifndef NOSE_HOOVER_THERMOSTATS_HPP
#define NOSE_HOOVER_THERMOSTATS_HPP

#include <torch/torch.h>

#include "Atoms.hpp"

class NoseHooverThermostats {
    public:
        //コンストラクタ
        NoseHooverThermostats(const torch::Tensor length, const torch::Tensor target_tmp, const torch::Tensor dof, const torch::Tensor tau, torch::Device device = torch::kCPU);
        NoseHooverThermostats();

        //ゲッター
        const torch::Tensor& length() const { return length_; }
        const torch::Tensor& positions() const { return positions_; }
        const torch::Tensor& masses() const { return masses_; }
        const torch::Tensor& velocities() const { return velocities_; }

        //更新
        void update(Atoms& atoms, torch::Tensor dt);
        
    private:
        //変数
        torch::Tensor length_;          //チェインの長さM (1, )
        torch::Tensor positions_;       //変位 (M, )
        torch::Tensor masses_;          //質量 (M, )
        torch::Tensor velocities_;      //速度 (M, )
        torch::Tensor forces_;          //力 (M, )

        torch::Tensor target_tmp_;      //目標温度 (1, )
        torch::Tensor dof_;             //自由度 (1, )

        torch::Tensor boltzmann_constant_;  //ボルツマン定数 (1, )
        torch::Device device_;              //計算デバイス
};

#endif
#ifndef NOSE_HOOVER_THERMOSTATS_HPP
#define NOSE_HOOVER_THERMOSTATS_HPP

#include <torch/torch.h>

class NoseHooverThermostats {
    public:
        //コンストラクタ
        NoseHooverThermostats(const torch::Tensor length, const torch::Tensor target_tmp, const torch::Tensor tau);

        //ゲッター
        const torch::Tensor& length() const { return length_; }
        const torch::Tensor& positions() const { return positions_; }
        const torch::Tensor& masses() const { return masses_; }
        const torch::Tensor& momentums() const { return momentums_; }

        //更新
        void update_positions();
        void update_momentums();

    private:
        torch::Tensor length_;          //チェインの長さM (1, )
        torch::Tensor positions;        //変位 (M, )
        torch::Tensor masses_;          //質量 (M, )
        torch::Tensor momentums_;       //運動量 (M, )

        torch::Tensor target_tmp;       //目標温度 (1, )
}

#endif
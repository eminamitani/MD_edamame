/**
* @file Atom.hpp
* @brief Atomクラス
*/

#ifndef ATOM_HPP
#define ATOM_HPP

#include <string>
#include <array>

#include <torch/script.h>
#include <torch/torch.h>

class Atom{
    public:
        //コンストラクタ
        Atom();
        Atom(std::string type, torch::Tensor position, torch::Tensor velocity, torch::Tensor force, torch::Device device);
        Atom(std::string type, std::array<double, 3>& position, std::array<double, 3>& velocity, std::array<double, 3>& force, torch::Device device);

        //ゲッタ
        /**
         * @brief 元素記号を取得します
         * @return 元素記号
         */
        const std::string& type() const { return type_; }
        /**
         * @brief 原子番号を取得します
         * @return 原子番号（torch::Tensor）
         */
        const torch::Tensor& atomic_number() const { return atomic_number_; }
        /**
         * @brief 原子の質量を取得します
         * @return 質量（torch::Tensor）
         */
        const torch::Tensor& mass() const { return mass_; }
        /**
         * @brief 原子の質量の逆数を取得します
         * @return 質量の逆数（torch::Tensor）
         */
        const torch::Tensor mass_inv() const;
        /**
         * @brief 原子の座標を取得します
         * @return 座標（torch::Tensor）
         */
        const torch::Tensor& position() const { return position_; }
        /**
         * @brief 原子の速度を取得します
         * @return 速度（torch::Tensor）
         */
        const torch::Tensor& velocity() const { return velocity_; }
        /**
         * @brief 原子にかかる力を取得します
         * @return 力（torch::Tensor）
         */
        const torch::Tensor& force() const { return force_; }
        /**
         * @brief デバイスを取得します
         * @return デバイス（torch::Tensor）
         */
        const torch::Device& device() const { return device_; }

        //セッタ
        /**
        * @brief 元素記号を指定します
        * @param[in] type 元素記号
        */
        void set_type(std::string& type);
        /**
        * @brief 原子の位置を指定します
        * @param[in] position 座標
        */        
        void set_position(torch::Tensor& position);
        /**
        * @brief 原子の速度を指定します
        * @param[in] velocity 速度
        */  
        void set_velocity(torch::Tensor& velocity);
        /**
        * @brief 原子にかかる力を指定します
        * @param[in] force 力
        */  
        void set_force(torch::Tensor& force);

       /**
        * @brief 原子の位置を指定します（std::array）
        * @param[in] position 座標
        */   
        void set_position(std::array<double, 3> position);
        /**
        * @brief 原子の速度を指定します（std::array）
        * @param[in] velocity 速度
        */  
        void set_velocity(std::array<double, 3> velocity);
        /**
        * @brief 原子にかかる力を指定します（std::array）
        * @param[in] force 力
        */  
        void set_force(std::array<double, 3> force);

       /**
        * @brief デバイスを移動させます
        * @param[in] device デバイス
        */  
        void to(torch::Device device);

        //その他
       /**
        * @brief 運動エネルギーを計算して、返します
        * @return 運動エネルギー
        */  
        torch::Tensor kinetic_energy();    //運動エネルギーを計算
    private:
        //原子種類
        std::string type_;
        torch::Tensor atomic_number_;
        torch::Tensor mass_;

        //プロパティ
        torch::Tensor position_;
        torch::Tensor velocity_;
        torch::Tensor force_;

        //計算デバイス
        torch::Device device_;
        torch::Tensor conversion_factor_;
};

#endif
/**
* @file NeighbourList.hpp
* @brief NeighbourListクラス
*/

#ifndef NEIGHBOUR_LIST_HPP
#define NEIGHBOUR_LIST_HPP

#include "Atoms.hpp"

#include <vector>

class NeighbourList {
    public:
        //コンストラクタ
        NeighbourList(torch::Tensor cutoff, torch::Tensor margin, torch::Device device = torch::kCPU);

        //ゲッタ
        /**
         * @brief ソース原子のインデックスを取得
         * @return ソース原子のインデックス
         * @note 戻り値は(num_edges, )のtorch::Tensor
         */
        const torch::Tensor& source_index() const { return source_index_; }
        /**
         * @brief ターゲット原子のインデックスを取得
         * @return ターゲット原子のインデックス
         * @note 戻り値は(num_edges, )のtorch::Tensor
         */
        const torch::Tensor& target_index() const { return target_index_; }
        /**
         * @brief カットオフ距離を取得
         * @return カットオフ距離
         * @note 戻り値は0次元のtorch::Tensor
         */
        const torch::Tensor& cutoff() const { return cutoff_; }
        /**
         * @brief 前回の原子配置を取得
         * @return カットオフ距離
         * @note 戻り値は(N, 3)のtorch::Tensor
         */
        const torch::Tensor& NL_config() const { return NL_config_; }
        /**
         * @brief デバイスを取得
         * @return デバイス
         */
        const torch::Device& device() const { return device_; }

        //デバイスの移動
        /**
         * @brief デバイスを移動
         * @param[in] device デバイス
         */
        void to(torch::Device device);

        //NLの作成
        /**
         * @brief 隣接リストを作成
         * @param[in] atoms 系
         */
        void generate(const Atoms& atoms);

        //NLの確認
        /**
         * @brief 隣接リストを確認し、必要があれば再作成
         * @param[in] atoms 系
         */
        void update(const Atoms& atoms);

    private:
    torch::Tensor source_index_;                     //ソース原子のインデックス (num_edges, )
    torch::Tensor target_index_;                     //ターゲット原子のインデックス (num_edges, )
    torch::Tensor NL_config_;                        //隣接リスト構築時点での配置を保存しておく配列
    torch::Tensor cutoff_;                           //カットオフ距離 (1, )
    torch::Tensor margin_;                           //カットオフからのマージン (1, )
    torch::Device device_;
};

#endif
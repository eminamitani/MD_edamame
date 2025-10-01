/**
* @file inference.hpp
* @brief NNPのための前処理と、モデルによる推論
*/

#ifndef INFERENCE_HPP
#define INFERENCE_HPP

#include "Atoms.hpp"
#include "NeighbourList.hpp"

#include <torch/script.h>
#include <torch/torch.h>

namespace inference{
    /**
     * @brief torchscript形式のモデルをロードします。
     * @return モデル
     * @param[in] model_path モデルのパス
     */
    torch::jit::script::Module load_model(std::string model_path);   
     /**
     * @brief 系の原子番号・接続情報から系のポテンシャル・力を推論します。
     * @return ポテンシャル・力
     * - `first` (torch::Tensor) ポテンシャル
     * 0次元のtorch::Tensor
     * - `second` (torch::Tensor) それぞれの原子が受ける力
     * (N, 3)のtorch::Tensor
     * @param[in] module NNPモデル
     * @param[in] x 原子番号
     * (N, )のtorch::Tensor
     * @param[in] edge_index グラフの接続行列
     * (num_edges, num_edges)のtorch::Tensor
     * @param[in] edge_weight 接続している原子同士の、原子間距離
     * (num_edges, )のtorch::Tensor
     */
    c10::ivalue::TupleElements infer_from_tensor(torch::jit::script::Module& module, torch::Tensor x, torch::Tensor edge_index, torch::Tensor edge_weight);                                                         
     /**
     * @brief 系の前処理をします。
     * 
     * 隣接リストを使用しない場合に用います。
     * 
     * @return 原子番号・接続情報
     * - `first` (torch::Tensor) 原子番号
     * (N, )のtorch::Tensor
     * - `second` (torch::Tensor) グラフの接続情報
     * (num_edges, num_edges)のtorch::Tensor
     * - `third` (torch::Tensor) 接続している原子同士の、原子間距離
     * (num_edges, )のtorch::Tensor
     * @param[in] atoms 前処理する系
     * @param[in] cutoff カットオフ距離
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RadiusInteractionGraph(Atoms& atoms, torch::Tensor cutoff);                                       //cutoff距離以内にある原子のペアを探す
     /**
     * @brief 系の前処理をします。
     * @return 原子番号・接続情報
     * - `first` (torch::Tensor) 原子番号
     * (N, )のtorch::Tensor  
     * - `second` (torch::Tensor) グラフの接続情報
     * (num_edges, num_edges)のtorch::Tensor
     * - `third` (torch::Tensor) 接続している原子同士の、原子間距離
     * (num_edges, )のtorch::Tensor
     * @param[in] atoms 前処理する系
     * @param[in] NL 隣接リスト
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RadiusInteractionGraph(Atoms& atoms, NeighbourList NL);                                           //隣接リストを使用する場合はこっち
     /**
     * @brief 系に対して、ポテンシャルと力を推論し、力とポテンシャルを系にセットします。
     * 隣接リストを使用しない場合に用います。
     * @param[in] module モデル
     * @param[in] atoms 系
     * @param[in] cutoff カットオフ距離
     */
    void calc_energy_and_force_MLP(torch::jit::script::Module& module, Atoms& atoms, torch::Tensor cutoff);                                                   //一つの構造に対して、エネルギーと力を計算
     /**
     * @brief 系に対して、ポテンシャルと力を推論し、力とポテンシャルを系にセットします。
     * @param[in] module モデル
     * @param[in] atoms 系
     * @param[in] NL 隣接リスト
     */
    void calc_energy_and_force_MLP(torch::jit::script::Module& module, Atoms& atoms, NeighbourList NL);
     /**
     * @brief 系に対して、ポテンシャルを推論し、力をその微分から計算します。その後、力とポテンシャルを系にセットします。
     * @note 使う必要はありません。
     * @param[in] module モデル
     * @param[in] atoms 系
     * @param[in] NL 隣接リスト
     */
    void infer_energy_with_MLP_and_clac_force(torch::jit::script::Module& module, Atoms& atoms, NeighbourList NL);
}

#endif
/**
* @file Atoms.hpp
* @brief Atomsクラス
*/

#ifndef ATOMS_HPP
#define ATOMS_HPP

#include "Atom.hpp"
#include "config.h"
#include <torch/torch.h>
#include <vector>

class Atoms {
public:
    // コンストラクタ
    Atoms(torch::Device device);
    Atoms(int N, torch::Device device);
    Atoms(std::vector<Atom> atoms, torch::Device device);

    // デバイス移動
    void to(torch::Device device);

    //ゲッタ
    /**
     * @brief すべての原子の座標を取得します
     * @return 座標
     * @note 戻り値は(N, 3)のtorch::Tensorです。
     */
    const torch::Tensor& positions() const { return positions_; }
    /**
     * @brief すべての原子の速度を取得します
     * @return 速度
     * @note 戻り値は(N, 3)のtorch::Tensorです。
     */
    const torch::Tensor& velocities() const { return velocities_; }
    /**
     * @brief すべての原子にかかっている力を取得します
     * @return 力
     * @note 戻り値は(N, 3)のtorch::Tensorです。
     */
    const torch::Tensor& forces() const { return forces_; }
    /**
     * @brief すべての原子の原子番号を取得します
     * @return 原子番号
     * @note 戻り値は(N, )のtorch::Tensorです。
     */
    const torch::Tensor& atomic_numbers() const { return atomic_numbers_; }
    /**
     * @brief すべての原子の質量を取得します
     * @return 質量
     * @note 戻り値は(N, )のtorch::Tensorです。
     */
    const torch::Tensor& masses() const { return masses_; }
    /**
     * @brief 系の1辺の長さを取得します
     * @return 1辺の長さ
     * @note 戻り値は0次元のtorch::Tensorです。正方形の系のみを想定しています。
     */
    const torch::Tensor& box_size() const { return box_size_; }
    /**
     * @brief デバイスを取得します
     * @return デバイス
     */
    const torch::Device& device() const { return device_; }
    /**
     * @brief 粒子数を取得します
     * @return 粒子数
     * @note 戻り値は0次元のtorch::Tensorです。
     */
    const torch::Tensor& size() const { return n_atoms_; }
    /**
     * @brief すべての原子の原子番号を取得します。
     * @return 原子番号
     * @note 戻り値は(N, )のstd::vector<std::string>です。
     */
    const std::vector<std::string>& types() const { return types_; }

    //セッタ
    /**
     * @brief すべての原子の座標を設定します。
     * 
     * 座標は(N, 3)のtorch::Tensorでなければなりません。
     * 
     * @param[in] positions 新しい座標
     */
    void set_positions(const torch::Tensor& positions);
    /**
     * @brief すべての原子の速度を設定します。
     * 
     * 速度は(N, 3)のtorch::Tensorでなければなりません。
     * 
     * @param[in] velocities 新しい速度
     */
    void set_velocities(const torch::Tensor& velocities);
    /**
     * @brief すべての原子にかかっている力を設定します。
     * 
     * 力は(N, 3)のtorch::Tensorでなければなりません。
     * 
     * @param[in] forces 新しい力
     */
    void set_forces(const torch::Tensor& forces);
    /**
     * @brief すべての原子の質量を設定します。
     * 
     * 質量は(N, )のtorch::Tensorでなければなりません。
     * 
     * @param[in] masses 新しい質量
     */
    void set_masses(const torch::Tensor& masses);
    /**
     * @brief 系のサイズを設定します。
     * 
     * 一辺の長さのみを指定します。正方形の系のみに対応しています。
     * 一辺の長さは0次元のtorch::Tensorでなければなりません。
     * 
     * @param[in] box_size 新しい系の一辺の長さ
     */
    void set_box_size(const torch::Tensor& box_size);
    /**
     * @brief ポテンシャルエネルギーを設定します。
     * 
     * ポテンシャルエネルギーは0次元のtorch::Tensorでなければなりません。
     * 
     * @param[in] potential_energy 新しいポテンシャル
     */
    void set_potential_energy(const torch::Tensor& potential_energy);
    /**
     * @brief すべての原子の原子番号を指定します。
     * 
     * 原子番号は(N, )のtorch::Tensorでなければなりません。
     * 
     * @param[in] atomic_numbers 新しい原子番号
     */
    void set_atomic_numbers(const torch::Tensor& atomic_numbers);
    /**
     * @brief すべての原子の元素記号を指定します。
     * 
     * 元素記号は(N, )のstd::vectorでなければなりません。
     * 
     * @param[in] types 新しい元素記号
     */
    void set_types(const std::vector<std::string>& types);

    //物理量の計算
    /**
     * @brief 運動エネルギーを計算して、返します。
     * @return 運動エネルギー
     * @note 戻り値は0次元のtorch::Tensorです。
     */
    torch::Tensor kinetic_energy() const;
    /**
     * @brief ポテンシャルエネルギーを取得します。
     * @return ポテンシャルエネルギー
     * @note 戻り値は0次元のtorch::Tensorです。
     */
    torch::Tensor potential_energy() const { return potential_energy_; }
    /**
     * @brief 系の温度を計算して、返します。
     * @return 温度
     * @note 戻り値は0次元のtorch::Tensorです。
     */
    torch::Tensor temperature() const;

    //その他
    /**
     * @brief 力に従って座標を更新します。
     * 
     * velocity-verlet法に基づいて更新します。
     * 
     * @param[in] dt 時間刻み幅
     * @param box シミュレーションボックスを何回はみ出したかを保存する配列
     */
    void positions_update(const torch::Tensor dt, torch::Tensor& box);
    /**
     * @brief 力に従って速度を更新します。
     * 
     * velocity-verlet法に基づいて更新します。
     * 
     * @param[in] dt 時間刻み幅
     */
    void velocities_update(const torch::Tensor dt);
    /**
     * @brief 周期境界条件の補正を適用します。
     */
    void apply_pbc(); //周期境界条件の補正
    /**
     * @brief 周期境界条件の補正を適用します。
     * 
     * @param シミュレーションボックスを何回はみ出したかを保存する配列
     */
    void apply_pbc(torch::Tensor& box);

     /**
     * @brief ドリフト速度を計算して、除去します。
     */
    void remove_drift();    //全体速度の除去

    //static関数
     /**
     * @brief 2粒子LJユニットを作成します。
     * @param[in] N 原子数
     * @param[in] ratio A粒子の割合
     * @param[in] rho 数密度
     * @param[in] device デバイス
     */
    static Atoms make_LJ_unit(const IntType N, const RealType ratio, const RealType rho, const torch::Device& device);  //LJユニットの作成
    
private:
    //計算デバイス
    torch::Device device_;

    //各原子のデータ
    torch::Tensor positions_;   //(num_atoms, 3)
    torch::Tensor velocities_;  //(num_atoms, 3)
    torch::Tensor forces_;      //(num_atoms, 3)
    torch::Tensor masses_;      //(num_atoms, )
    torch::Tensor atomic_numbers_;  //(num_atoms, )
    std::vector<std::string> types_;

    //系のデータ
    torch::Tensor n_atoms_;
    torch::Tensor potential_energy_;
    torch::Tensor box_size_;

    //定数
    torch::Tensor conversion_factor_;
    torch::Tensor boltzmann_constant_;
};

#endif
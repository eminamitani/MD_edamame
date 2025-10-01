/**
* @file LJ.hpp
* @brief LJポテンシャルによる力とポテンシャルの計算
* @note NNPの方と無理やりくっつけているので使用非推奨です。テストのためだけに使っています。
*/

#ifndef LJ_HPP
#define LJ_HPP

#include "config.h"
#include "Atoms.hpp"
#include "NeighbourList.hpp"

namespace LJ {
    inline const torch::Tensor MBLJ_sij1 = torch::tensor({
        {1.0, 0.8},
        {0.8, 0.88}
    });

    inline const torch::Tensor MBLJ_energy = torch::tensor({
        {1.0, 1.5},
        {1.5, 0.5}
    });

    // LJポテンシャルとその一階微分
    torch::Tensor LJpotential(const torch::Tensor distances, const torch::Tensor sigmas);
    torch::Tensor deriv_1st_LJpotential(const torch::Tensor distances, const torch::Tensor sigmas);

    //力とエネルギーの計算
    void calc_force(Atoms& atoms, NeighbourList NL);
    void calc_potential(Atoms& atoms, NeighbourList NL);

    void calc_energy_and_force(Atoms& atoms, NeighbourList NL);
}

#endif
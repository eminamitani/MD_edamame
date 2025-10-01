#include "LJ.hpp"

torch::Tensor LJ::LJpotential(const torch::Tensor distances, const torch::Tensor sigmas) {
    const torch::Tensor dist6 = torch::pow(distances, 6);
    const torch::Tensor sigma6 = torch::pow(sigmas, 6);
    return 4.0 * sigma6 * (sigma6 - dist6)/(dist6 * dist6);
}

torch::Tensor LJ::deriv_1st_LJpotential(const torch::Tensor distances, const torch::Tensor sigmas) {
    const torch::Tensor dist6 = torch::pow(distances, 6);
    const torch::Tensor sigma6 = torch::pow(sigmas, 6);
    return - 24.0 / distances * sigma6 * (2.0 * sigma6 - dist6)/(dist6 * dist6);
}

void LJ::calc_force(Atoms& atoms, NeighbourList NL) {
    const auto device = atoms.device();

    const torch::Tensor& pos = atoms.positions();
    const torch::Tensor& Lbox = atoms.box_size();
    const torch::Tensor Linv = 1 / Lbox;

    //NLから読み込んだインデックスの原子の距離を計算
    const torch::Tensor source_pos = pos.index({NL.source_index()});
    const torch::Tensor target_pos = pos.index({NL.target_index()});

    torch::Tensor diff_pos_vec = source_pos - target_pos;
    diff_pos_vec -= Lbox * torch::floor(diff_pos_vec * Linv + 0.5);

    // 距離の2乗を計算
    torch::Tensor dist2 = torch::sum(diff_pos_vec.pow(2), 1);

    // 種類の判定のために、フィルタリング前の全ペアの原子番号を取得
    torch::Tensor atomic_numbers = atoms.atomic_numbers();
    torch::Tensor source_atomic_numbers_all = atomic_numbers.index({NL.source_index()});
    torch::Tensor target_atomic_numbers_all = atomic_numbers.index({NL.target_index()});

    // ペアごとのカットオフ距離を計算 (同種なら1.5, 異種なら2.0)
    torch::Tensor same_type_mask = (source_atomic_numbers_all == target_atomic_numbers_all);
    torch::Tensor cutoffs = torch::where(same_type_mask, 1.5, 2.0).to(device);
    torch::Tensor cutoff2s = cutoffs.pow(2);

    // ペアごとのカットオフ距離でフィルタリング
    torch::Tensor mask = torch::lt(dist2, cutoff2s); // dist2 < cutoff^2

    // マスクを適用
    torch::Tensor source_index = NL.source_index().index({mask});
    torch::Tensor target_index = NL.target_index().index({mask});
    dist2 = dist2.index({mask});
    diff_pos_vec = diff_pos_vec.index({mask});
    cutoffs = cutoffs.index({mask}); // 使用するカットオフ値もフィルタリング

    // フィルタリング後のペアの原子番号
    torch::Tensor source_atomic_numbers = source_atomic_numbers_all.index({mask});
    torch::Tensor target_atomic_numbers = target_atomic_numbers_all.index({mask});

    //力の計算
    torch::Tensor dist = torch::sqrt(dist2);
    torch::Tensor sigmas = MBLJ_sij1.to(device).index({source_atomic_numbers, target_atomic_numbers});
    torch::Tensor epsilons = MBLJ_energy.to(device).index({source_atomic_numbers, target_atomic_numbers});

    torch::Tensor deriv_1st = deriv_1st_LJpotential(dist, sigmas) - deriv_1st_LJpotential(cutoffs, sigmas);
    deriv_1st *= epsilons;

    torch::Tensor force_scalar = - deriv_1st / dist;
    torch::Tensor force_vec = force_scalar.unsqueeze(1) * diff_pos_vec;

    //加算
    torch::Tensor total_forces = torch::zeros_like(pos);

    total_forces.index_add_(0, source_index, force_vec);
    total_forces.index_add_(0, target_index, -force_vec);

    //力をセット
    atoms.set_forces(total_forces / 2.0);
}

void LJ::calc_potential(Atoms& atoms, NeighbourList NL) {
    const auto device = atoms.device();
    const torch::Tensor& pos = atoms.positions();
    const torch::Tensor& Lbox = atoms.box_size();
    const torch::Tensor Linv = 1 / Lbox;

    //NLから読み込んだインデックスの原子の距離を計算
    const torch::Tensor source_pos = pos.index({NL.source_index()});
    const torch::Tensor target_pos = pos.index({NL.target_index()});

    torch::Tensor diff_pos_vec = source_pos - target_pos;
    diff_pos_vec -= Lbox * torch::floor(diff_pos_vec * Linv + 0.5);

    // 距離の2乗を計算
    torch::Tensor dist2 = torch::sum(diff_pos_vec.pow(2), 1);

    // 種類の判定のために、フィルタリング前の全ペアの原子番号を取得
    torch::Tensor atomic_numbers = atoms.atomic_numbers();
    torch::Tensor source_atomic_numbers_all = atomic_numbers.index({NL.source_index()});
    torch::Tensor target_atomic_numbers_all = atomic_numbers.index({NL.target_index()});

    // ペアごとのカットオフ距離を計算 (同種なら1.5, 異種なら2.0)
    torch::Tensor same_type_mask = (source_atomic_numbers_all == target_atomic_numbers_all);
    torch::Tensor cutoffs = torch::where(same_type_mask, 1.5, 2.0).to(device);
    torch::Tensor cutoff2s = cutoffs.pow(2);

    // ペアごとのカットオフ距離でフィルタリング
    torch::Tensor mask = torch::lt(dist2, cutoff2s); // dist2 < cutoff^2

    // マスクを適用
    torch::Tensor source_index = NL.source_index().index({mask});
    torch::Tensor target_index = NL.target_index().index({mask});
    dist2 = dist2.index({mask});
    diff_pos_vec = diff_pos_vec.index({mask});
    cutoffs = cutoffs.index({mask}); // 使用するカットオフ値もフィルタリング

    // フィルタリング後のペアの原子番号
    torch::Tensor source_atomic_numbers = source_atomic_numbers_all.index({mask});
    torch::Tensor target_atomic_numbers = target_atomic_numbers_all.index({mask});

    //ポテンシャルの計算
    torch::Tensor dist = torch::sqrt(dist2);
    torch::Tensor sigmas = MBLJ_sij1.to(device).index({source_atomic_numbers, target_atomic_numbers});
    torch::Tensor epsilons = MBLJ_energy.to(device).index({source_atomic_numbers, target_atomic_numbers});

    torch::Tensor potentials = LJpotential(dist, sigmas) - LJpotential(cutoffs, sigmas) - deriv_1st_LJpotential(cutoffs, sigmas) * (dist - cutoffs);
    potentials *= epsilons;

    torch::Tensor potential = torch::sum(potentials);

    //ポテンシャルをセット
    atoms.set_potential_energy(potential / 2.0);
}

void LJ::calc_energy_and_force(Atoms& atoms, NeighbourList NL) {
    LJ::calc_force(atoms, NL);
    LJ::calc_potential(atoms, NL);
}
#include "NoseHooverThermostats.hpp"
#include "config.h"

#include <stdexcept>

NoseHooverThermostats::NoseHooverThermostats(const IntType length, const torch::Tensor target_tmp, const torch::Tensor tau, torch::Device device) : 
length_(length), target_tmp_(target_tmp.to(device)), tau_(tau.to(device)), device_(device)
{
    torch::TensorOptions option = torch::TensorOptions().dtype(kRealType).device(device_);

    //更新関数の初期化
    if(length_ == 1) {
        update_function = [this](torch::Tensor& atoms_velocities, const torch::Tensor& kinetic_energy, const torch::Tensor& dt) {
            NHC1(atoms_velocities, kinetic_energy, dt);
        };
    }
    else if (length_ == 2) {
        update_function = [this](torch::Tensor& atoms_velocities, const torch::Tensor& kinetic_energy, const torch::Tensor& dt) {
            NHC2(atoms_velocities, kinetic_energy, dt);
        };
    }
    else {
        update_function = [this](torch::Tensor& atoms_velocities, const torch::Tensor& kinetic_energy, const torch::Tensor& dt) {
            NHCM(atoms_velocities, kinetic_energy, dt);
        };
    }

    //変数の初期化
    dof_ = torch::tensor(0, torch::TensorOptions().device(device).dtype(kIntType));
    positions_ = torch::zeros({length_}, option);
    velocities_ = torch::zeros({length_}, option);
    forces_ = torch::zeros({length_}, option);
    masses_ = torch::zeros({length_}, option);

    //定数
    boltzmann_constant_ = torch::tensor(boltzmann_constant, option);
}

NoseHooverThermostats::NoseHooverThermostats() : NoseHooverThermostats(1.0, torch::tensor(300.0), torch::tensor(1.0), torch::kCPU) {}

void NoseHooverThermostats::setup(Atoms& atoms) {
    //質量の初期化
    dof_ = torch::tensor(3 * atoms.size().item<IntType>() - 3);
    torch::Tensor tau2 = torch::pow(tau_, 2);
    masses_.fill_(boltzmann_constant_ * target_tmp_ * tau2);
    masses_[0] *= dof_;
}

void NoseHooverThermostats::setup(const torch::Tensor dof) {
    dof_ = dof;
    torch::Tensor tau2 = torch::pow(tau_, 2);
    masses_.fill_(boltzmann_constant_ * target_tmp_ * tau2);
    masses_[0] *= dof_;
}

void NoseHooverThermostats::update(Atoms& atoms, const torch::Tensor& dt) {
    torch::Tensor kinetic_energy = atoms.kinetic_energy().detach().clone();
    torch::Tensor atoms_velocities = atoms.velocities().detach().clone();
    update(atoms_velocities, kinetic_energy, dt);
    atoms.set_velocities(atoms_velocities);
}

void NoseHooverThermostats::update(torch::Tensor& atoms_velocities, const torch::Tensor& kinetic_energy, const torch::Tensor& dt) {
    update_function(atoms_velocities, kinetic_energy, dt);
}

//1段
void NoseHooverThermostats::NHC1(torch::Tensor& atoms_velocities, const torch::Tensor& kinetic_energy, const torch::Tensor& dt) {
    torch::Tensor AKIN = 2 * kinetic_energy;
    torch::Tensor scale = torch::tensor(1.0, torch::TensorOptions().dtype(kRealType).device(device_));
    //逆順の更新
    forces_[0] = (AKIN - (dof_ * target_tmp_ * boltzmann_constant_)) / masses_[0];
    velocities_[0] += dt * 0.25 * forces_[0];
    //スケーリング
    scale *= torch::exp(- 0.5 * dt * velocities_[0]);
    AKIN *= torch::exp(- dt * velocities_[0]);
    //変位の更新
    positions_ += dt * 0.5 * velocities_;
    //順方向の更新
    forces_[0] = (AKIN - (dof_ * target_tmp_ * boltzmann_constant_)) / masses_[0];
    velocities_[0] += dt * 0.25 * forces_[0];
    //速度のスケーリング
    atoms_velocities *= scale;
}

//2段
void NoseHooverThermostats::NHC2(torch::Tensor& atoms_velocities, const torch::Tensor& kinetic_energy, const torch::Tensor& dt) {
    torch::Tensor AKIN = 2 * kinetic_energy;
    torch::Tensor scale = torch::tensor(1.0, torch::TensorOptions().dtype(kRealType).device(device_));
    //逆順の更新
    forces_[1] = ((masses_[0] * torch::pow(velocities_[0], 2)) - (boltzmann_constant_ * target_tmp_)) / masses_[1];
    velocities_[1] += dt * 0.25 * forces_[1];
    velocities_[0] *= torch::exp(- 0.125 * dt * velocities_[1]);
    forces_[0] = (AKIN - (dof_ * target_tmp_ * boltzmann_constant_)) / masses_[0];
    velocities_[0] += 0.25 * dt * forces_[0];
    velocities_[0] *= torch::exp(- 0.125 * dt * velocities_[1]);
    //スケーリング
    scale *= torch::exp(- 0.5 * dt * velocities_[0]);
    AKIN *= torch::exp(- dt * velocities_[0]);
    //変位の更新
    positions_ += dt * 0.5 * velocities_;
    //順方向の更新
    velocities_[0] *= torch::exp(- 0.125 * dt * velocities_[1]);
    forces_[0] = (AKIN - (dof_ * target_tmp_ * boltzmann_constant_)) / masses_[0];
    velocities_[0] += 0.25 * dt * forces_[0];
    velocities_[0] *= torch::exp(- 0.125 * dt * velocities_[1]);
    forces_[1] = ((masses_[0] * torch::pow(velocities_[0], 2)) - (boltzmann_constant_ * target_tmp_)) / masses_[1];
    velocities_[1] += dt * 0.25 * forces_[1];
    //速度のスケーリング
    atoms_velocities *= scale;
}

//M段（一応。テストあまりしていないです。）
void NoseHooverThermostats::NHCM(torch::Tensor& atoms_velocities, const torch::Tensor& kinetic_energy, const torch::Tensor& dt) {
    torch::Tensor scale = torch::tensor(1.0, torch::TensorOptions().dtype(kRealType).device(device_));
    torch::Tensor Akin = 2 * kinetic_energy;

    //逆順ループ
    forces_[length_ - 1] = (masses_[length_ - 2] * torch::pow(velocities_[length_ - 2], 2) - boltzmann_constant_ * target_tmp_) / masses_[length_ - 1];
    velocities_[length_ - 1] += dt * 0.25 * forces_[length_ - 1];
    for(int64_t i = length_ - 2; i > 0; i --) {
        velocities_[i] *= torch::exp(- velocities_[i + 1] * dt * 0.125);
        forces_[i] = (masses_[i - 1] * torch::pow(velocities_[i - 1], 2) - boltzmann_constant_ * target_tmp_) / masses_[i];
        velocities_[i] += dt * 0.25 * forces_[i];
        velocities_[i] *= torch::exp(- velocities_[i + 1] * dt * 0.125);
    }
    velocities_[0] *= torch::exp(- velocities_[1] * dt * 0.125);
    forces_[0] = (Akin - dof_ * boltzmann_constant_ * target_tmp_) / masses_[0];
    velocities_[0] += dt * 0.25 * forces_[0];
    velocities_[0] *= torch::exp(- velocities_[1] * dt * 0.125);

    //スケーリング
    scale *= torch::exp(- velocities_[0] * dt * 0.5);
    Akin *= torch::exp(- dt * velocities_[0]);

    //変位の更新
    positions_ += 0.5 * dt * velocities_;

    //順方向ループ
    velocities_[0] *= torch::exp(- velocities_[1] * dt * 0.125);
    forces_[0] = (Akin - dof_ * boltzmann_constant_ * target_tmp_) / masses_[0];
    velocities_[0] += dt * 0.25 * forces_[0];
    velocities_[0] *= torch::exp(- velocities_[1] * dt * 0.125);
    for(int64_t i = 1; i < length_ - 1; i ++) {
        velocities_[i] *= torch::exp(- velocities_[i + 1] * dt * 0.125);
        forces_[i] = (masses_[i - 1] * torch::pow(velocities_[i - 1], 2) - boltzmann_constant_ * target_tmp_) / masses_[i];
        velocities_[i] += dt * 0.25 * forces_[i];
        velocities_[i] *= torch::exp(- velocities_[i + 1] * dt * 0.125);
    }
    forces_[length_ - 1] = (masses_[length_ - 2] * torch::pow(velocities_[length_ - 2], 2) - boltzmann_constant_ * target_tmp_) / masses_[length_ - 1];
    velocities_[length_ - 1] += dt * 0.25 * forces_[length_ - 1];

    atoms_velocities *= scale;
}
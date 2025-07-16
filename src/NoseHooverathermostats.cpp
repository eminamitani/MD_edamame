#include "NoseHooverThermostats.hpp"
#include "Atoms.hpp"
#include "config.h"
#include "inference.hpp"

NoseHooverThermostats::NoseHooverThermostats(const torch::Tensor length, const torch::Tensor target_tmp, const torch::Tensor dof, const torch::Tensor tau, torch::Device device) : length_(length), target_tmp_(target_tmp), dof_(dof), device_(device)  
{   
    IntType length_int = length_.item<IntType>();
    boltzmann_constant_ = torch::tensor(boltzmann_constant, torch::TensorOptions().dtype(kRealType).device(device_));

    //変位、速度、力の初期化
    positions_ = torch::zeros({length_int}, device);
    velocities_ = torch::zeros({length_int}, device);
    forces_ = torch::zeros({length_int}, device);

    //質量の初期化
    masses_ = torch::empty({length_int});
    torch::Tensor tau_2 = torch::pow(tau, 2);
    masses_.fill_(boltzmann_constant_ * target_tmp_ * tau_2);
    masses_[0] *= dof;
}

NoseHooverThermostats::NoseHooverThermostats() : NoseHooverThermostats(torch::tensor(1.0), torch::tensor(0.0), torch::tensor(0.0), torch::tensor(0.0), torch::kCPU) {}

void NoseHooverThermostats::update(Atoms& atoms, torch::Tensor dt) {
    torch::Tensor scale = torch::tensor(1.0, torch::TensorOptions().dtype(kRealType).device(device_));
    IntType length_int = length_.item<IntType>();
    torch::Tensor Akin = 2 * atoms.kinetic_energy();

    //逆順ループ
    forces_[length_int - 1] = (masses_[length_int - 2] * torch::pow(velocities_[length_int - 2], 2) - boltzmann_constant_ * target_tmp_) / masses_[length_int - 1];
    velocities_[length_int - 1] += dt * 0.25 * forces_[length_int - 1];
    for(int64_t i = length_int - 2; i > 0; i --) {
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
    for(int64_t i = 1; i < length_int - 1; i ++) {
        velocities_[i] *= torch::exp(- velocities_[i + 1] * dt * 0.125);
        forces_[i] = (masses_[i - 1] * torch::pow(velocities_[i - 1], 2) - boltzmann_constant_ * target_tmp_) / masses_[i];
        velocities_[i] += dt * 0.25 * forces_[i];
        velocities_[i] *= torch::exp(- velocities_[i + 1] * dt * 0.125);
    }
    forces_[length_int - 1] = (masses_[length_int - 2] * torch::pow(velocities_[length_int - 2], 2) - boltzmann_constant_ * target_tmp_) / masses_[length_int - 1];
    velocities_[length_int - 1] += dt * 0.25 * forces_[length_int - 1];

    atoms.set_velocities(atoms.velocities() * scale);
}
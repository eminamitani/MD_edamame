#include "BussiThermostat.hpp"

#include <iostream>
#include <random>

//コンストラクタ
BussiThermostat::BussiThermostat(const torch::Tensor targ_temp, const torch::Tensor tau, torch::Device device) : targ_temp_(targ_temp), tau_(tau), device_(device), mt_(std::random_device()())
{
    torch::TensorOptions option = torch::TensorOptions().device(device);
    dof_ = torch::empty({}, option.dtype(kIntType));
    boltzmann_constant_ = torch::tensor(boltzmann_constant, option.dtype(kRealType));
}

BussiThermostat::BussiThermostat(const RealType targ_temp, const RealType tau, torch::Device device) : BussiThermostat(torch::tensor(targ_temp, device), torch::tensor(tau, device), device) {}


//セットアップ
void BussiThermostat::setup(const Atoms& atoms) {
    setup(3 * atoms.size());
}

void BussiThermostat::setup(const torch::Tensor& dof) {
    dof_ = dof;
    IntType dof_int = dof_.item<IntType>();
    if(dof_int % 2 == 0) {
        RealType alpha = (dof_int - 1) / 2;
        dist_ = std::gamma_distribution<>(alpha, 1);
    }
    else {
        RealType alpha = (dof_int - 2) / 2;
        dist_ = std::gamma_distribution<>(alpha, 1);
    }
}

//更新
void BussiThermostat::update(Atoms& atoms, const torch::Tensor& dt) {
    torch::Tensor atoms_velocities = atoms.velocities().clone();
    torch::Tensor kinetic_energy = atoms.kinetic_energy().clone();
    update(atoms_velocities, kinetic_energy, dt);
    atoms.set_velocities(atoms_velocities);
}

void BussiThermostat::update(torch::Tensor& atoms_velocities, const torch::Tensor& kinetic_energy, const torch::Tensor& dt) {
    //乱数の作成
    torch::Tensor rand = torch::randn({2});
    torch::Tensor rand2 = torch::pow(rand, 2);
    torch::Tensor gamma = torch::tensor(dist_(mt_), torch::TensorOptions().device(device_).dtype(kRealType));

    //目標運動エネルギー
    torch::Tensor targ_kin = (dof_ * boltzmann_constant_ * targ_temp_) / 2;

    //スケーリング要素の計算
    torch::Tensor f = torch::exp(- dt / tau_);
    torch::Tensor alpha2 = f + (targ_kin * (1 - f) * (rand2[0] + rand2[1] + 2 * gamma)) / (dof_ * kinetic_energy) + 2 * rand[0] * torch::sqrt((f * (1 - f) * targ_kin) / (dof_ * kinetic_energy));

    //スケーリング
    atoms_velocities *= torch::sqrt(alpha2);
}
#include "BussiThermostat.hpp"

#include <iostream>
#include <random>

//コンストラクタ
BussiThermostat::BussiThermostat(const torch::Tensor& targ_temp, const torch::Tensor& tau, const torch::Device& device) : targ_temp_(targ_temp), tau_(tau), device_(device), boltzmann_constant_(torch::tensor(boltzmann_constant, kRealType)) {
    targ_temp_ = targ_temp_.to(device);
    tau_ = tau_.to(device);
    boltzmann_constant_ = boltzmann_constant_.to(device);
}

BussiThermostat::BussiThermostat(const RealType& targ_temp, const RealType& tau, const torch::Device& device) : BussiThermostat(torch::tensor(targ_temp), torch::tensor(tau), device) {}

//セットアップ
void BussiThermostat::setup(const Atoms& atoms) {
    setup(3 * atoms.size());
}

void BussiThermostat::setup(const torch::Tensor& dof) {
    dof_ = dof;
    dof_ = dof_.to(device_);
}

//更新
void BussiThermostat::update(Atoms& atoms, const torch::Tensor& dt) {
    torch::Tensor atoms_velocities = atoms.velocities().clone();
    torch::Tensor kinetic_energy = atoms.kinetic_energy().clone();
    update(atoms_velocities, kinetic_energy, dt);
    atoms.set_velocities(atoms_velocities);
}

void BussiThermostat::update(torch::Tensor& atoms_velocities, const torch::Tensor& kinetic_energy, const torch::Tensor& dt) {
    //乱数の生成
    torch::Tensor rand = torch::randn({dof_.item<IntType>()}, torch::TensorOptions().dtype(kRealType).device(device_)); //標準正規分布に従うdof個の独立な乱数
    torch::Tensor r = rand[0];
    torch::Tensor rand2 = torch::sum(torch::pow(rand, 2));

    //目標運動エネルギー
    torch::Tensor targ_kin = (dof_ * boltzmann_constant_ * targ_temp_) / 2;

    //スケーリング要素を計算
    torch::Tensor f = torch::exp(- dt / tau_);
    torch::Tensor alpha2 = f + (targ_kin * (1 - f) * rand2) / (dof_ * kinetic_energy) + 2 * r * torch::sqrt((targ_kin * f * (1 - f)) / (dof_ * kinetic_energy));

    //スケーリング
    atoms_velocities *= torch::sqrt(alpha2);
}

void BussiThermostat::set_temp(const RealType& targ_temp){
    targ_temp_ = torch::tensor(targ_temp, device_);
}
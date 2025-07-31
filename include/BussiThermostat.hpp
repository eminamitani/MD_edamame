#ifndef BUSSI_THERMOSTAT_HPP
#define BUSSI_THERMOSTAT_HPP

#include "Atoms.hpp"
#include "config.h"

#include <functional>
#include <random>

#include <torch/torch.h>

class BussiThermostat {
    public:
        BussiThermostat(const torch::Tensor& targ_temp, const torch::Tensor& tau, torch::Device& device);
        BussiThermostat(const RealType& targ_temp, const RealType& tau, torch::Device& device);

        void setup(const Atoms& atoms);
        void setup(const torch::Tensor& dof);

        void update(Atoms& atoms, const torch::Tensor& dt);
        void update(torch::Tensor& atoms_velocities, const torch::Tensor& kinetic_energy, const torch::Tensor& dt);

    private:
        torch::Tensor dof_;
        torch::Tensor tau_;
        torch::Tensor targ_temp_;

        torch::Tensor boltzmann_constant_;

        torch::Device device_;
};

#endif
#ifndef BUSSI_THERMOSTAT_HPP
#define BUSSI_THERMOSTAT_HPP

#include "Atoms.hpp"
#include "config.h"

#include <functional>
#include <random>

#include <torch/torch.h>

class BussiThermostat {
    public:
        BussiThermostat(const torch::Tensor& targ_temp, const torch::Tensor& tau, const torch::Device& device = torch::kCPU);
        BussiThermostat(const RealType& targ_temp, const RealType& tau, const torch::Device& device = torch::kCPU);

        const torch::Tensor& temp() const { return targ_temp_; }
        RealType temp_real() const { return targ_temp_.item<RealType>(); }

        void setup(const Atoms& atoms);
        void setup(const torch::Tensor& dof);

        void update(Atoms& atoms, const torch::Tensor& dt);
        void update(torch::Tensor& atoms_velocities, const torch::Tensor& kinetic_energy, const torch::Tensor& dt);

        void set_temp(const torch::Tensor& targ_temp) { targ_temp_ = targ_temp; }
        void set_temp(const RealType& targ_temp);

    private:
        torch::Tensor dof_;
        torch::Tensor tau_;
        torch::Tensor targ_temp_;

        torch::Tensor boltzmann_constant_;

        torch::Device device_;
};

#endif
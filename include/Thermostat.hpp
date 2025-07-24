#ifndef THERMOSTAT_HPP
#define THERMOSTAT_HPP

#include "Atoms.hpp"
#include <torch/torch.h>

//熱浴の抽象クラス
class Thermostat {
    public:
        virtual void update(Atoms& atoms, const torch::Tensor& dt) = 0;
        virtual void update(torch::Tensor& atoms_velocities, const torch::Tensor& kinetic_energy, const torch::Tensor& dt) = 0;
        virtual void setup(Atoms& atoms) = 0;
        virtual ~Thermostat() = default;
};


#endif
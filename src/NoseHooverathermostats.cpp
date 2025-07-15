#include "NoseHooverThermostats.hpp"

NoseHooverThermostats::NoseHooverThermostats(const torch::Tensor length, const torch::Tensor target_tmp, const torch::Tensor tau) : length_(length) 
{
    positions_ = torch::zeros({length_});
    momentums_ = torch::zeros({length_});
}
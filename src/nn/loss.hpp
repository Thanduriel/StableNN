#pragma once

#include <functional>
#include <torch/torch.h>
#include "hyperparam.hpp"

namespace nn {

	// network output, target, data
	using LossFn = std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&)>;

	torch::Tensor lpLoss(const torch::Tensor& input, const torch::Tensor& target, c10::Scalar p);
	torch::Tensor energyLoss(const torch::Tensor& netInput, const torch::Tensor& netOutput);

	// @param _train loss function for training which includes regularization terms
	LossFn makeLossFunction(const HyperParams& _params, bool _train = true);
}
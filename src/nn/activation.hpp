#pragma once

#include <functional>
#include <torch/torch.h>

namespace nn {

	using ActivationFn = std::function<torch::Tensor(const torch::Tensor&)>;

	inline torch::Tensor identity(const torch::Tensor& x) { return x; }

	// odd sigmoid: f(x) = -f(-x)
	inline torch::Tensor zerosigmoid(const torch::Tensor& x) { return torch::sigmoid(x) - 0.5; }
}
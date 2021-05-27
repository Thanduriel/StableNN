#pragma once

#include <functional>
#include <torch/torch.h>

namespace nn {

	using ActivationFn = std::function<torch::Tensor(const torch::Tensor&)>;

	std::ostream& operator<<(std::ostream& _out, const ActivationFn& _activation);
	std::istream& operator>>(std::istream& _in, ActivationFn& _activation);

	inline torch::Tensor identity(const torch::Tensor& x) { return x; }

	// odd sigmoid: f(x) = -f(-x); this is basically just tanh :)
	inline torch::Tensor zerosigmoid(const torch::Tensor& x) { return torch::sigmoid(x) - 0.5; }

	// fixed parameters so that it can be serialized
	inline torch::Tensor elu(const torch::Tensor& x) { return torch::elu(x); }

	inline torch::Tensor cubicrelu(const torch::Tensor& x) { return torch::relu(torch::pow(x, 3)); }
}
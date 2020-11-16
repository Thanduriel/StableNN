#pragma once

#include <functional>
#include <torch/torch.h>

namespace nn {

	using ActivationFn = std::function<torch::Tensor(const torch::Tensor&)>;
	// torch::Tensor(*)(const torch::Tensor&);
	// helpers for serialization
	int64_t toIndex(ActivationFn _fn);
	ActivationFn fromIndex(int64_t _fn);
}
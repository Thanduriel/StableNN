#pragma once

#include <torch/torch.h>
#include <memory>

namespace nn {

	template<typename Module>
	Module clone(const Module& _module)
	{
		using ModuleImpl = typename Module::Impl;
		return Module(std::dynamic_pointer_cast<ModuleImpl>(_module->clone()));
	}

	inline torch::Tensor lp_loss(const torch::Tensor& input, const torch::Tensor& target, c10::Scalar p)
	{
		return (input - target).norm(p, {1}).mean();
	}
}
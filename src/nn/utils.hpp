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

	// shift entries left in a tensor (batch size x time series) and adds _newEntry to the end.
	torch::Tensor shiftTimeSeries(const torch::Tensor& _old, const torch::Tensor& _newEntry, int _stateSize);

	torch::Tensor lp_loss(const torch::Tensor& input, const torch::Tensor& target, c10::Scalar p);

	void exportTensor(const torch::Tensor& _tensor, const std::string& _fileName);
}
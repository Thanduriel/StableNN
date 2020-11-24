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
	inline torch::Tensor shiftTimeSeries(const torch::Tensor& _old, const torch::Tensor& _newEntry, int _stateSize)
	{
		using namespace torch::indexing;
		const int64_t len = _old.size(1);
		torch::Tensor newInput = torch::zeros_like(_old);
		newInput.index_put_({ "...", Slice(0, len - _stateSize) },
			_old.index({ "...", Slice(_stateSize, len) }));
		newInput.index_put_({ "...", Slice(len - _stateSize, len) }, _newEntry);

		return newInput;
	}

	inline torch::Tensor lp_loss(const torch::Tensor& input, const torch::Tensor& target, c10::Scalar p)
	{
		return (input - target).norm(p, {1}).mean();
	}
}
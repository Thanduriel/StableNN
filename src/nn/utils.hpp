#pragma once

#include "../systems/state.hpp"
#include <torch/torch.h>
#include <c10/core/ScalarType.h>
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

	template<typename T, size_t N>
	torch::Tensor arrayToTensor(const std::array<T, N>& _data, const c10::TensorOptions& _options = c10::TensorOptions(c10::CppTypeToScalarType<T>()))
	{
		return torch::from_blob(const_cast<T*>(_data.data()),
			{ static_cast<int64_t>(N) },
			_options);
	}

	template<typename T, size_t N>
	std::array<T,N> tensorToArray(const torch::Tensor& _tensor)
	{
		return *reinterpret_cast<std::array<T, N>*>(_tensor.data_ptr<T>());
	}

	// Functor which converts an array of system states into a tensor
	struct StateToTensor
	{
		template<typename System>
		torch::Tensor operator()(const System&,
			const typename System::State* _state,
			int64_t _numStates,
			int64_t _batchSize,
			const c10::TensorOptions& _options) const
		{
			assert(_numStates % _batchSize == 0);
			constexpr int64_t stateSize = systems::sizeOfState<System>();

			return torch::from_blob(const_cast<typename System::State*>(_state),
				{ _batchSize, stateSize * _numStates / _batchSize },
				_options);
		}
	};
}
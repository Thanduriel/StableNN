#pragma once

#include "hyperparam.hpp"
#include "nnmaker.hpp"
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

	template<typename Module>
	int64_t countParams(const Module& _module)
	{
		int64_t numParams = 0;
		for (auto& tensor : _module->parameters())
		{
			if (tensor.requires_grad())
				numParams += tensor.numel();
		}

		return numParams;
	}

	template<typename Module>
	void save(const Module& _module, const HyperParams& _params)
	{
		const auto name = _params.get<std::string>("name", "net");
		torch::save(_module,  name + ".pt");
		std::ofstream file(name + ".hparam");
		file << _params;
	}

	// @param _name File name without ending.
	template<typename NetType, bool UseWrapper>
	auto load(const HyperParams& _params, const std::string& _name = "", torch::Device _device = torch::kCPU,
		HyperParams* _outParams = nullptr)
	{
		const std::string& name = _name.empty() ? *_params.get<std::string>("name") : _name;

		HyperParams params = _params;
		std::ifstream file(name + ".hparam");
		file >> params;
		if (_outParams) *_outParams = params;

		const std::string storedType = params.get<std::string>("net_type", "unknown");
		const std::string type = nn::sanitizeString(std::string(typeid(NetType).name()));
		if (storedType != type)
		{
			std::cout << "[Warning] Possible model type mismatch. Expected " << type
				<< " but saved model is of type " << storedType << ".\n";
		}

		auto net = nn::makeNetwork<NetType, UseWrapper>(params);
		torch::load(net, name + ".pt", _device);

		return net;
	}

	// Shift entries left in a tensor (batch size x time series) and adds _newEntry to the end.
	torch::Tensor shiftTimeSeries(const torch::Tensor& _old, const torch::Tensor& _newEntry, int _stateSize);

	// @param _pgfPlotsFormat write explicit row and column index with just one value per line
	void exportTensor(const torch::Tensor& _tensor, const std::string& _fileName, bool _pgfPlotsFormat = false);

	template<typename T, size_t N>
	torch::Tensor arrayToTensor(const std::array<T, N>& _data, const c10::TensorOptions& _options = c10::TensorOptions(c10::CppTypeToScalarType<T>()))
	{
		return torch::from_blob(const_cast<T*>(_data.data()),
			{ static_cast<int64_t>(N) },
			_options).clone();
	}

	template<typename T, size_t N>
	std::array<T,N> tensorToArray(const torch::Tensor& _tensor)
	{
		return *reinterpret_cast<std::array<T, N>*>(_tensor.data_ptr<T>());
	}

	// Functor which converts an array of system states into a 2-d tensor
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

	// Similar to StateToTensor, but introduces an explicit time dimension
	// @param TimeLast if true, the last dimension is time, otherwise it is the state
	template<bool TimeLast = true>
	struct StateToTensorTimeseries
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

			torch::Tensor tensor = torch::from_blob(const_cast<typename System::State*>(_state),
				{ _batchSize, _numStates / _batchSize, stateSize },
				_options);
			if constexpr (TimeLast)
				return tensor.permute({ 0,2,1 });
			else
				return tensor;
		}
	};

	// Specialize this for a network architecture that requires different inputs.
	template<typename Network>
	struct InputMakerSelector
	{
		using type = StateToTensor;
	};

	template<typename Network>
	using MakeTensor_t = typename InputMakerSelector<Network>::type;


	template<typename Network>
	struct IdentityMap
	{
		static torch::Tensor forward(const torch::Tensor& _input)
		{
			return _input;
		}
	};
}
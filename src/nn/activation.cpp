#include "activation.hpp"

namespace nn {

	using ActivationFnType = torch::Tensor(const torch::Tensor&);

	const std::unordered_map< ActivationFnType*, std::string > ACTIVATION_NAMES = {
		{torch::tanh, "tanh"},
		{torch::relu, "relu"},
		{torch::sigmoid, "sigmoid"},
		{identity, "identity"},
		{zerosigmoid, "zerosigmoid"},
	};

	std::ostream& operator<<(std::ostream& _out, const ActivationFn& _activation)
	{
		auto it = ACTIVATION_NAMES.find(*_activation.target<ActivationFnType*>());
		if (it != ACTIVATION_NAMES.end())
			_out << it->second;
		else
			_out << _activation.target_type().name();

		return _out;
	}

/*	int64_t toIndex(ActivationFn _fn)
	{
		auto it = std::find(ACTIVATIONS.begin(), ACTIVATIONS.end(), _fn);
		assert(it != ACTIVATIONS.end());

		return std::distance(ACTIVATIONS.begin(), it);
	}

	ActivationFn fromIndex(int64_t _fn)
	{
		return ACTIVATIONS[_fn];
	}*/
}
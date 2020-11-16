#include "activation.hpp"

namespace nn {

/*	constexpr std::array< ActivationFn, 3 > ACTIVATIONS = {
		torch::tanh,
		torch::relu,
		torch::sigmoid,
	};

	int64_t toIndex(ActivationFn _fn)
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
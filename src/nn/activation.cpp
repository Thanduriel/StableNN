#include "activation.hpp"

namespace nn {

	using ActivationFnType = torch::Tensor(const torch::Tensor&);

	const std::unordered_map< ActivationFnType*, std::string > ACTIVATION_NAMES = {
		{torch::tanh, "tanh"},
		{torch::relu, "relu"},
		{torch::sigmoid, "sigmoid"},
		{identity, "identity"},
		{zerosigmoid, "zerosigmoid"},
		{elu, "elu"},
		{cubicrelu, "cubicrelu"}
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

	std::istream& operator>>(std::istream& _in, ActivationFn& _activation)
	{
		std::string name;
		_in >> name;

		auto it = std::find_if(ACTIVATION_NAMES.begin(), ACTIVATION_NAMES.end(), [&](const auto& pair) 
			{
				return pair.second == name;
			});

		if (it != ACTIVATION_NAMES.end())
			_activation = *it->first;
		else
			std::cerr << "Unknown activation " << name << std::endl;

		return _in;
	}
}
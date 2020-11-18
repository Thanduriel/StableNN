#include "antisymmetric.hpp"

namespace nn {
	using namespace torch;

	AntiSymmetricCellImpl::AntiSymmetricCellImpl(int64_t _size, double _gamma, bool _bias)
		: size(_size), gamma(_gamma), useBias(_bias)
	{
		reset();
	}

	void AntiSymmetricCellImpl::reset()
	{
		weight = register_parameter("weight",
			torch::empty({ size, size }));
		if (useBias) {
			bias = register_parameter("bias", torch::empty(size));
		}
		else {
			bias = register_parameter("bias", {}, false);
		}
		diffusion = register_parameter("diffusion", torch::eye(size) * gamma, false);

		reset_parameters();
	}

	void AntiSymmetricCellImpl::reset_parameters()
	{
		torch::nn::init::kaiming_uniform_(weight, std::sqrt(5));
		if (bias.defined()) 
		{
			auto& [fan_in, fan_out] =
				torch::nn::init::_calculate_fan_in_and_fan_out(weight);
			const auto bound = 1 / std::sqrt(fan_in);
			torch::nn::init::uniform_(bias, -bound, bound);
		}
	}

	void AntiSymmetricCellImpl::pretty_print(std::ostream& stream) const {
		stream << std::boolalpha
			<< "nn::AntiSymmetric(size=" << size
			<< ", gamma=" << gamma
			<< ", bias=" << useBias << ")";
	}

	Tensor AntiSymmetricCellImpl::forward(const Tensor& input)
	{
		return torch::nn::functional::linear(input, system_matrix(), bias);
	}

	Tensor AntiSymmetricCellImpl::system_matrix() const
	{
		return 0.5 * (weight - weight.t() - diffusion);
	}

	// ********************************************************* //
	AntiSymmetricImpl::AntiSymmetricImpl(int64_t _inputs,
		int64_t _hiddenLayers,
		double _diffusion,
		double _totalTime,
		bool _useBias,
		ActivationFn _activation)
		: timeStep(_totalTime / _hiddenLayers), activation(std::move(_activation))
	{
		layers.reserve(_hiddenLayers);

		for (int64_t i = 0; i < _hiddenLayers; ++i)
		{
			layers.emplace_back(_inputs, _diffusion, _useBias);
			register_module("layer" + std::to_string(i), layers.back());
		}
	}

	torch::Tensor AntiSymmetricImpl::forward(torch::Tensor x)
	{
		for (auto& layer : layers)
			x = x + timeStep * activation(layer(x));
		return x;
	}


}
#include "antisymmetric.hpp"

namespace nn {
	using namespace torch;

	AntiSymmetricImpl::AntiSymmetricImpl(int64_t _size, double _gamma, bool _bias)
		: size(_size), gamma(_gamma), useBias(_bias)
	{
		reset();
	}

	void AntiSymmetricImpl::reset()
	{
		weight = register_parameter("weight",
			torch::empty({ size, size }));
		if (useBias) {
			bias = register_parameter("bias", torch::empty(size));
		}
		else {
			bias = register_parameter("bias", {}, false);
		}
		diffusion = register_parameter("diffusion", torch::eye(size) * gamma);

		reset_parameters();
	}

	void AntiSymmetricImpl::reset_parameters()
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

	void AntiSymmetricImpl::pretty_print(std::ostream& stream) const {
		stream << std::boolalpha
			<< "nn::AntiSymmetric(size=" << size
			<< ", gamma=" << gamma
			<< ", bias=" << useBias << ")";
	}

	Tensor AntiSymmetricImpl::forward(const Tensor& input)
	{
		return torch::nn::functional::linear(input, 0.5 * (weight - weight.transpose(0,1) - diffusion), bias);
	}

	// ********************************************************* //
	AntiSymmetricNet::AntiSymmetricNet(int64_t _inputs,
		int64_t _hiddenLayers,
		double _diffusion,
		double _totalTime,
		bool _useBias)
		: timeStep(_totalTime / _hiddenLayers)
	{
		hiddenLayers.reserve(_hiddenLayers);

		for (int64_t i = 0; i < _hiddenLayers; ++i)
		{
			hiddenLayers.emplace_back(_inputs, _diffusion);
			register_module("hidden" + std::to_string(i), hiddenLayers.back());
		}
	}

	torch::Tensor AntiSymmetricNet::forward(torch::Tensor x)
	{
		for (auto& layer : hiddenLayers)
			x = x + timeStep * torch::tanh(layer(x));
		return x;
	}


}
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
			const auto& [fan_in, fan_out] =
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
		return torch::nn::functional::linear(input, systemMatrix(), bias);
	}

	Tensor AntiSymmetricCellImpl::systemMatrix() const
	{
		return 0.5 * (weight - weight.t() - diffusion);
	}

	// ********************************************************* //
	AntiSymmetricImpl::AntiSymmetricImpl(const AntiSymmetricOptions& _options)
		: options(_options)
	{
		reset();
	}

	void AntiSymmetricImpl::reset()
	{
		layers.clear();
		layers.reserve(options.num_layers());

		timeStep = options.total_time() / options.num_layers();
		for (int64_t i = 0; i < options.num_layers(); ++i)
		{
			layers.emplace_back(options.input_size(), options.diffusion(), options.bias());
			register_module("layer" + std::to_string(i), layers.back());
		}
	}

	torch::Tensor AntiSymmetricImpl::forward(torch::Tensor x)
	{
		auto& activation = options.activation();

		for (auto& layer : layers)
			x = x + timeStep * activation(layer(x));
		return x;

/*		Tensor x0 = x;
		Tensor x1 = x0 + 0.5 * timeStep * activation(layers[0](x));
		bool flipped = false;

		for (auto it = layers.begin() + 1; it != layers.end(); ++it)
		{
			auto& layer = *it;
			Tensor& y0 = flipped ? x1 : x0;
			Tensor& y1 = flipped ? x0 : x1;
			y0 = 2.0 * y1 - y0 + timeStep * activation(layer(y1));
			flipped = !flipped;
		}

		return flipped ? x0 : x1;*/
	}


}
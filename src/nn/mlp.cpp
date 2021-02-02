#include "mlp.hpp"

namespace nn {

	MultiLayerPerceptronImpl::MultiLayerPerceptronImpl(const MLPOptions& _options)
		: options(_options)
	{
		reset();
	}

	void MultiLayerPerceptronImpl::reset()
	{
		torch::nn::LinearOptions hiddenOptions(options.hidden_size(), options.hidden_size());
		hiddenOptions.bias(options.bias());
		layers.clear();
		layers.reserve(options.hidden_layers());

		for (int64_t i = 0; i < options.hidden_layers(); ++i)
		{
			layers.emplace_back(torch::nn::Linear(hiddenOptions));
			register_module("hidden" + std::to_string(i), layers.back());
		}
	}



	torch::Tensor MultiLayerPerceptronImpl::forward(torch::Tensor x)
	{
		auto& activation = options.activation();
		for (auto& layer : layers)
			x = x + activation(layer(x));

		return x;
	}

	// ****************************************************************** //
	MultiLayerPerceptronExtImpl::MultiLayerPerceptronExtImpl(const MLPOptions& _options)
		: options(_options)
	{
		reset();
	}

	void MultiLayerPerceptronExtImpl::reset()
	{
		torch::nn::LinearOptions hiddenOptions(options.hidden_size(), options.hidden_size());
		hiddenOptions.bias(options.bias());
		layers.clear();
		layers.reserve(options.hidden_layers());

		for (int64_t i = 0; i < options.hidden_layers(); ++i)
		{
			const std::string suffix = std::to_string(i);
			layers.emplace_back(torch::nn::Linear(hiddenOptions));
			register_module("hidden" + suffix, layers.back());

			idScales.emplace_back(register_parameter("id" + suffix, torch::ones({ 1 })));
			torch::nn::init::uniform_(idScales.back(), 0.1, 2.0);
			addScales.emplace_back(register_parameter("add" + suffix, torch::empty({ 1 })));
			torch::nn::init::uniform_(addScales.back(), -1.0, 1.0);
		}
	}



	torch::Tensor MultiLayerPerceptronExtImpl::forward(torch::Tensor x)
	{
		for (size_t i = 0; i < layers.size(); ++i)
		{
			x = idScales[i] * x + addScales[i] * torch::tanh(layers[i](x));
		}
	
		return x;
	}

}
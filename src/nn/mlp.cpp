#include "mlp.hpp"

namespace nn {

	MultiLayerPerceptronImpl::MultiLayerPerceptronImpl(const MLPOptions& _options)
		: options(_options),
		inputLayer(nullptr),
		outputLayer(nullptr)
	{
		reset();
	}

	void MultiLayerPerceptronImpl::reset()
	{
		inputLayer = torch::nn::Linear(torch::nn::LinearOptions(
			options.input_size(), 
			options.hidden_layers() ? options.hidden_size() : options.output_size()).bias(options.bias()));
		register_module("input", inputLayer);

		torch::nn::LinearOptions hiddenOptions(options.hidden_size(), options.hidden_size());
		hiddenOptions.bias(options.bias());
		hiddenLayers.clear();
		hiddenLayers.reserve(options.hidden_layers());

		for (int64_t i = 0; i < options.hidden_layers(); ++i)
		{
			hiddenLayers.emplace_back(torch::nn::Linear(hiddenOptions));
			register_module("hidden" + std::to_string(i), hiddenLayers.back());
		}

		outputLayer = torch::nn::Linear(torch::nn::LinearOptions(
			options.hidden_layers() ? options.hidden_size() : options.input_size(),
			options.output_size()).bias(options.bias()));
		register_module("output", outputLayer);
	}



	torch::Tensor MultiLayerPerceptronImpl::forward(torch::Tensor x)
	{
		x = x + torch::tanh(inputLayer(x));
		for (auto& layer : hiddenLayers)
			x = x + torch::tanh(layer(x));

		x = outputLayer(x);
		return x;
	}


}
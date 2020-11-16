#include "mlp.hpp"

namespace nn {

	MultiLayerPerceptronImpl::MultiLayerPerceptronImpl(
		int64_t _inputs, int64_t _outputs, int64_t _hiddenLayerSize, int64_t _hiddenLayers, bool _useBias)
		: inputLayer(torch::nn::LinearOptions(_inputs, _hiddenLayers ? _hiddenLayerSize : _outputs).bias(_useBias)),
		outputLayer(torch::nn::LinearOptions(_hiddenLayers ? _hiddenLayerSize : _inputs, _outputs).bias(_useBias))
	{
		register_module("input", inputLayer);
		
		torch::nn::LinearOptions options(_hiddenLayerSize, _hiddenLayerSize);
		options.bias(_useBias);

		for (int64_t i = 0; i < _hiddenLayers; ++i)
		{
		//	hiddenLayers->push_back(torch::nn::Linear(options));
			hiddenLayers.emplace_back(torch::nn::Linear(options));
			register_module("hidden" + std::to_string(i), hiddenLayers.back());
		}
	//	register_module("hidden", hiddenLayers);
		register_module("output", outputLayer);
	}

	torch::Tensor MultiLayerPerceptronImpl::forward(torch::Tensor x)
	{
		x = torch::tanh(inputLayer(x));
		for (auto& layer : hiddenLayers)
			x = x + torch::tanh(layer(x));
		//	x = x + torch::tanh(layer->as<torch::nn::Linear>()->forward(x));
		x = outputLayer(x);
		return x;
	}


}
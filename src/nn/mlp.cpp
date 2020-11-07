#include "mlp.hpp"

namespace nn {

	MultiLayerPerceptron::MultiLayerPerceptron(int64_t _inputs, int64_t _outputs, int64_t _hiddenLayerSize, int64_t _hiddenLayers)
		: inputLayer(_inputs, _hiddenLayers ? _hiddenLayerSize : _outputs),
		outputLayer(_hiddenLayers ? _hiddenLayerSize : _inputs, _outputs)
	{
		register_module("input", inputLayer);
		hiddenLayers.reserve(_hiddenLayers);
		for (int64_t i = 0; i < _hiddenLayers; ++i)
		{
			hiddenLayers.emplace_back(_hiddenLayerSize, _hiddenLayerSize);
			register_module("hidden" + std::to_string(i), hiddenLayers.back());
		}
		register_module("output", outputLayer);
	}

	torch::Tensor MultiLayerPerceptron::forward(torch::Tensor x)
	{
	//	x = /*torch::sigmoid*/(inputLayer(x));
		for (auto& layer : hiddenLayers)
			x = torch::sigmoid(layer(x));
	//	x = outputLayer(x);
		return x;
	}


}
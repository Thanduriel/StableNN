#pragma once

#include <torch/torch.h>

namespace nn {

	struct MultiLayerPerceptron : torch::nn::Module
	{
		MultiLayerPerceptron(int64_t _inputs = 1, int64_t _outputs = 1, 
			int64_t _hiddenLayerSize = 32,
			int64_t _hiddenLayers = 1,
			bool _useBias = false);

		torch::Tensor forward(torch::Tensor _input);

		torch::nn::Linear inputLayer;
		std::vector<torch::nn::Linear> hiddenLayers;
		torch::nn::Linear outputLayer;
	};
}
#pragma once

#include <torch/torch.h>

namespace nn {

	struct MultiLayerPerceptronImpl : public torch::nn::Module
	{
		MultiLayerPerceptronImpl(int64_t _inputs = 1, int64_t _outputs = 1,
			int64_t _hiddenLayerSize = 32,
			int64_t _hiddenLayers = 1,
			bool _useBias = false);

		torch::Tensor forward(torch::Tensor _input);

		torch::nn::Linear inputLayer;
		//torch::nn::ModuleList hiddenLayers;
		std::vector<torch::nn::Linear> hiddenLayers;
		torch::nn::Linear outputLayer;
	};

	using MultiLayerPerceptron = MultiLayerPerceptronImpl;
	//TORCH_MODULE(MultiLayerPerceptron);
}
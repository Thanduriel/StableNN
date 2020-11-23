#pragma once

#include "activation.hpp"
#include <torch/torch.h>

namespace nn {

	struct MLPOptions
	{
		MLPOptions(int64_t _inputSize) : input_size_(_inputSize), output_size_(_inputSize), hidden_size_(_inputSize) {}

		TORCH_ARG(int64_t, input_size);
		TORCH_ARG(int64_t, output_size);
		TORCH_ARG(int64_t, hidden_size);
		TORCH_ARG(int64_t, hidden_layers) = 0;
		TORCH_ARG(bool, bias) = true;
		TORCH_ARG(ActivationFn, activation) = torch::tanh;
	};

	struct MultiLayerPerceptronImpl : public torch::nn::Cloneable<MultiLayerPerceptronImpl>
	{
		explicit MultiLayerPerceptronImpl(const MLPOptions& _options);

		void reset() override;

		torch::Tensor forward(torch::Tensor _input);

		MLPOptions options;

		torch::nn::Linear inputLayer;
		//torch::nn::ModuleList hiddenLayers;
		std::vector<torch::nn::Linear> hiddenLayers;
		torch::nn::Linear outputLayer;
	};

	//using MultiLayerPerceptron = MultiLayerPerceptronImpl;
	TORCH_MODULE(MultiLayerPerceptron);
}
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
		using Options = MLPOptions;
		explicit MultiLayerPerceptronImpl(const MLPOptions& _options);

		void reset() override;

		torch::Tensor forward(torch::Tensor _input);

		MLPOptions options;

		torch::nn::Linear inputLayer;
		std::vector<torch::nn::Linear> hiddenLayers;
		torch::nn::Linear outputLayer;
	};

	TORCH_MODULE(MultiLayerPerceptron);

	struct MultiLayerPerceptronExtImpl : public torch::nn::Cloneable<MultiLayerPerceptronExtImpl>
	{
		using Options = MLPOptions;
		explicit MultiLayerPerceptronExtImpl(const MLPOptions& _options);

		void reset() override;

		torch::Tensor forward(torch::Tensor _input);

		MLPOptions options;

		std::vector<torch::nn::Linear> hiddenLayers;
		std::vector<torch::Tensor> idScales;
		std::vector<torch::Tensor> addScales;
	};

	TORCH_MODULE(MultiLayerPerceptronExt);
}
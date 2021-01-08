#pragma once

#include "activation.hpp"
#include <torch/torch.h>

namespace nn {

	struct ConvolutionalOptions
	{
		ConvolutionalOptions(int64_t _inputSize, int64_t _filterSize) 
			: input_size_(_inputSize), filter_size_(_filterSize) {}

		TORCH_ARG(int64_t, input_size);
		TORCH_ARG(int64_t, num_layers) = 1;
		TORCH_ARG(bool, bias) = false;
		TORCH_ARG(int64_t, filter_size);
		TORCH_ARG(ActivationFn, activation) = torch::tanh;
	};

	struct ConvolutionalImpl : public torch::nn::Cloneable<ConvolutionalImpl>
	{
		using Options = ConvolutionalOptions;
		explicit ConvolutionalImpl(const Options& _options);

		void reset() override;

		torch::Tensor forward(torch::Tensor _input);

		Options options;
		std::vector<torch::nn::Conv1d> layers;
	};

	TORCH_MODULE(Convolutional);

}
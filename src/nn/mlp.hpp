#pragma once

#include <torch/torch.h>

namespace nn {

	struct MultiLayerPerceptron : torch::nn::Module
	{
		MultiLayerPerceptron(int64_t _inputs = 1, int64_t _outputs = 1);

		torch::Tensor forward(torch::Tensor _input);

		torch::nn::Linear linear1;
		torch::nn::Linear linear2;
		torch::nn::Linear linear3;
	};
}
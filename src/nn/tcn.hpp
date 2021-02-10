#pragma once

#include "activation.hpp"
#include "utils.hpp"

namespace nn {

	struct TCNOptions
	{
		TCNOptions(int64_t in_channels, int64_t out_channels, int64_t window_size);

		TORCH_ARG(int64_t, in_channels);
		TORCH_ARG(int64_t, hidden_channels);
		TORCH_ARG(int64_t, out_channels);
		TORCH_ARG(int64_t, window_size);
		TORCH_ARG(int64_t, residual_blocks) = 1;
		TORCH_ARG(int64_t, block_size) = 2;
		TORCH_ARG(int64_t, kernel_size) = 3;
		TORCH_ARG(bool, bias) = false;
		TORCH_ARG(ActivationFn, activation) = torch::tanh;

		// amount of dropout after convolutional layers
		TORCH_ARG(double, dropout) = 0.0;

		// uses stride instead of dilation to increase the effective window size
		// adds AvgPool to residual connections to resize the output
		TORCH_ARG(bool, average) = false;

		// have residual connections between blocks
		TORCH_ARG(bool, residual) = true;
	};

	struct TCNImpl : public torch::nn::Cloneable<TCNImpl>
	{
		using Options = TCNOptions;

		explicit TCNImpl(const Options& options);

		void reset() override;

		torch::Tensor forward(torch::Tensor x);

		torch::nn::Sequential layers;
		TCNOptions options;
	};

	TORCH_MODULE(TCN);

	template<>
	struct InputMakerSelector<TCN>
	{
		using type = StateToTensorTimeseries<true>;
	};
}
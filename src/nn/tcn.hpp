#pragma once

#include "activation.hpp"
#include "utils.hpp"

namespace nn {

	template<size_t D>
	struct TCNOptions
	{
		TCNOptions(int64_t in_channels, int64_t out_channels, int64_t window_size, torch::ExpandingArray<D> kernel_size)
			: in_channels_(in_channels),
			hidden_channels_(std::max(in_channels / 2, out_channels)),
			out_channels_(out_channels),
			window_size_(window_size),
			kernel_size_(kernel_size)
		{
		}

		TORCH_ARG(int64_t, in_channels);
		TORCH_ARG(int64_t, hidden_channels);
		TORCH_ARG(int64_t, out_channels);
		TORCH_ARG(int64_t, window_size);
		TORCH_ARG(int64_t, residual_blocks) = 1;
		TORCH_ARG(int64_t, block_size) = 2;
		TORCH_ARG(torch::ExpandingArray<D>, kernel_size);
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

	template<size_t D, typename Derived>
	class TCNImpl : public torch::nn::Cloneable<Derived>
	{
	public:
		using Options = TCNOptions<D>;

		explicit TCNImpl(const Options& options);

		void reset() override;

		torch::Tensor forward(torch::Tensor x);

		torch::nn::Sequential layers;
		TCNOptions<D> options;
	};

	class TCN1DImpl : public TCNImpl<1, TCN1DImpl>
	{
	public:
		using TCNImpl<1, TCN1DImpl>::TCNImpl;
	};

	TORCH_MODULE(TCN1D);

	using TCN = TCN1D;

	template<>
	struct InputMakerSelector<TCN>
	{
		using type = StateToTensorTimeseries<true>;
	};
}
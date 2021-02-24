#pragma once

#include "activation.hpp"
#include "utils.hpp"

namespace nn {

	template<size_t D>
	struct TemporalConvBlockOptions
	{
		TemporalConvBlockOptions(int64_t in_channels, int64_t out_channels, torch::ExpandingArray<D> kernel_size)
			: in_channels_(in_channels),
			out_channels_(out_channels),
			kernel_size_(kernel_size)
		{
		}

		TORCH_ARG(int64_t, in_channels);
		TORCH_ARG(int64_t, out_channels);
		TORCH_ARG(int64_t, block_size) = 2;
		TORCH_ARG(torch::ExpandingArray<D>, kernel_size);
		TORCH_ARG(int64_t, dilation) = 1;
		TORCH_ARG(bool, bias) = false;
		TORCH_ARG(ActivationFn, activation) = torch::tanh;

		// amount of dropout after convolution layers
		TORCH_ARG(double, dropout) = 0.0;

		// uses stride instead of dilation to increase the effective window size
		// adds AvgPool to residual connections to resize the output
		TORCH_ARG(bool, average) = false;

		// have residual connections between blocks
		TORCH_ARG(bool, residual) = true;
	};

	template<int64_t D, typename Derived = void>
	class TemporalConvBlockImpl : public torch::nn::Cloneable< 
		std::conditional_t<std::is_same_v<Derived, void>, TemporalConvBlockImpl<D>, Derived>>
	{
	public:
		using Options = TemporalConvBlockOptions<D>;
		explicit TemporalConvBlockImpl(const Options& _options);
		void reset() override;
		torch::Tensor forward(torch::Tensor x);

		using Conv = std::conditional_t<D == 1, torch::nn::Conv1d, torch::nn::Conv2d>;
		using AvgPool = std::conditional_t<D == 1, torch::nn::AvgPool1d, torch::nn::AvgPool2d>;

		std::vector<Conv> layers;
		Conv residual;
		AvgPool avg_residual;
		torch::nn::Dropout dropout_layer; // todo: also consider Dropout2D ?
		TemporalConvBlockOptions<D> options;
	};

	class TemporalConvBlock1DImpl : public TemporalConvBlockImpl<1, TemporalConvBlock1DImpl>
	{
	public:
		using TemporalConvBlockImpl<1, TemporalConvBlock1DImpl>::TemporalConvBlockImpl;
	};
	TORCH_MODULE(TemporalConvBlock1D);

	class TemporalConvBlock2DImpl : public TemporalConvBlockImpl<2, TemporalConvBlock2DImpl>
	{
	public:
		using TemporalConvBlockImpl<2, TemporalConvBlock2DImpl>::TemporalConvBlockImpl;
	};
	TORCH_MODULE(TemporalConvBlock2D);

	template<size_t D>
	struct TCNOptions
	{
		// @param in_size (channels, time, spatial...) 
		// @param out_size (channels, spatial...)
		// in and out size are required to compute the proper output transform
		// for now spatial dimensions should match
		TCNOptions(torch::ExpandingArray<D+1> in_size, torch::ExpandingArray<D> out_size, torch::ExpandingArray<D> kernel_size)
			: in_size_(in_size),
			out_size_(out_size),
			hidden_channels_(std::max(in_size->front() / 2, out_size->front())),
			kernel_size_(kernel_size)
		{
		}

		TORCH_ARG(torch::ExpandingArray<D+1>, in_size);
		TORCH_ARG(torch::ExpandingArray<D>, out_size);
		TORCH_ARG(int64_t, hidden_channels);
		TORCH_ARG(int64_t, residual_blocks) = 1;

		// forwarded to TemporalConvBlockOptions
		TORCH_ARG(torch::ExpandingArray<D>, kernel_size);
		TORCH_ARG(bool, bias) = false;
		TORCH_ARG(ActivationFn, activation) = torch::tanh;
		TORCH_ARG(int64_t, block_size) = 2;
		TORCH_ARG(double, dropout) = 0.0;
		TORCH_ARG(bool, average) = false;
		TORCH_ARG(bool, residual) = true;
	};

	// Temporal convolution network with residual blocks and exponential dilation.
	template<size_t D, typename Derived>
	class TCNImpl : public torch::nn::Cloneable<Derived>
	{
	public:
		using Options = TCNOptions<D>;

		explicit TCNImpl(const Options& options);

		void reset() override;

		torch::Tensor forward(torch::Tensor x);

		using TCNBlock = TemporalConvBlockImpl<D>;

		torch::nn::Sequential layers;
		TCNOptions<D> options;
	};

	class TCN1DImpl : public TCNImpl<1, TCN1DImpl>
	{
	public:
		using TCNImpl<1, TCN1DImpl>::TCNImpl;
	};
	TORCH_MODULE(TCN1D);

	class TCN2DImpl : public TCNImpl<2, TCN2DImpl>
	{
	public:
		using TCNImpl<2, TCN2DImpl>::TCNImpl;
	};
	TORCH_MODULE(TCN2D);

	// variant with an extra residual connection that adds the most recent time-step to the result
	class SimpleTCNImpl : public TCNImpl<1, SimpleTCNImpl>
	{
	public:
		using TCNImpl<1, SimpleTCNImpl>::TCNImpl;

		torch::Tensor forward(torch::Tensor x);
	};
	TORCH_MODULE(SimpleTCN);
	using TCN = SimpleTCN;

	template<>
	struct InputMakerSelector<TCN>
	{
		using type = StateToTensorTimeseries<true>;
	};
}
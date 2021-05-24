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
		TORCH_ARG(int64_t, dilation) = 1; // only applied to temporal dimension

		TORCH_ARG(torch::ExpandingArray<D>, kernel_size);
		TORCH_ARG(bool, bias) = false;
		TORCH_ARG(ActivationFn, activation) = torch::tanh;

		// number of serial layers in this block
		TORCH_ARG(int64_t, block_size) = 2;

		// only the first value is used unless interleaved == true
		using padding_mode_t = torch::ExpandingArray<D,torch::nn::detail::conv_padding_mode_t>;
		TORCH_ARG(padding_mode_t, padding_mode) = padding_mode_t(torch::kZeros);

		// amount of dropout after convolution layers
		TORCH_ARG(double, dropout) = 0.0;

		// uses stride instead of dilation to increase the effective time window size
		// adds AvgPool to residual connections to resize the output
		TORCH_ARG(bool, average) = false;

		// have a residual connection from input to output
		// if in_channels != out_channels, a convolution with kernel size 1 is added
		TORCH_ARG(bool, residual) = true;

		// only for D == 2: use 1d kernels on alternating dimensions instead of the full kernel_size
		TORCH_ARG(bool, interleaved) = false;
	};

	// Stack of convolution layers with optional dropout and residual connection
	template<int64_t D>
	class TemporalConvBlockImpl : public torch::nn::Cloneable<TemporalConvBlockImpl<D>> // std::conditional_t<std::is_same_v<Derived, void>, TemporalConvBlockImpl<D>, Derived>
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

	using TemporalConvBlock1dImpl = TemporalConvBlockImpl<1>;
	TORCH_MODULE(TemporalConvBlock1d);
	using TemporalConvBlock2dImpl = TemporalConvBlockImpl<2>;
	TORCH_MODULE(TemporalConvBlock2d);

	// ============================================================================

	template<size_t D>
	struct TCNOptions
	{
		// @param in_size (channels, time, spatial...) 
		// @param out_size (channels, spatial...)
		// in and out size are required to compute the proper output transform
		// size for spatial dimensions are currently ignored
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

		// see TemporalConvBlockOptions
		TORCH_ARG(torch::ExpandingArray<D>, kernel_size);
		TORCH_ARG(bool, bias) = false;
		TORCH_ARG(ActivationFn, activation) = torch::tanh;
		TORCH_ARG(int64_t, block_size) = 2;
		TORCH_ARG(double, dropout) = 0.0;
		TORCH_ARG(bool, average) = false;
		TORCH_ARG(bool, residual) = true;
		using padding_mode_t = torch::ExpandingArray<D, torch::nn::detail::conv_padding_mode_t>;
		TORCH_ARG(padding_mode_t, padding_mode) = padding_mode_t(torch::kZeros);
		TORCH_ARG(bool, interleaved) = false;
	};

	// Generic temporal convolution network with residual blocks and exponential dilation.
	template<size_t D>
	class TCNImpl : public torch::nn::Cloneable<TCNImpl<D>>
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

	using TCN1dImpl = TCNImpl<1>;
	TORCH_MODULE(TCN1d);
	using TCN2dImpl = TCNImpl<2>;
	TORCH_MODULE(TCN2d);

	// variant with an extra residual connection that adds the most recent time-step to the result
	class SimpleTCNImpl : public torch::nn::Cloneable<SimpleTCNImpl>
	{
	public:
		using Options = TCNOptions<1>;
		explicit SimpleTCNImpl(const Options& options);

		void reset() override;
		torch::Tensor forward(torch::Tensor x);

		TCN1d layers;
		Options options;
	};
	TORCH_MODULE(SimpleTCN);
	using TCN = SimpleTCN;

	template<>
	struct InputMakerSelector<SimpleTCN>
	{
		using type = StateToTensorTimeseries<true>;
	};

	// ============================================================================

	// 2d TCN with experimental extensions
	struct ExtTCNOptions : public TCNOptions<2>
	{
		using TCNOptions<2>::TCNOptions;

		TORCH_ARG(int64_t, ext_residual) = false;
	//	TORCH_ARG(int64_t, symmetric) = false;
	};

	class ExtTCNImpl : public torch::nn::Cloneable<ExtTCNImpl>
	{
	public:
		using Options = ExtTCNOptions;
		explicit ExtTCNImpl(const Options& options);

		void reset() override;
		torch::Tensor forward(torch::Tensor x);

		TCN2d layers;
		Options options;
	};
	TORCH_MODULE(ExtTCN);
}
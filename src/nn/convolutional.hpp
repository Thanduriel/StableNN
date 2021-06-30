#pragma once

#include "activation.hpp"
#include "extconv.hpp"
#include "utils.hpp"
#include <torch/torch.h>

namespace nn {

	struct ConvolutionalOptions
	{
		ConvolutionalOptions(int64_t _numChannels, int64_t _filterSize) 
			: num_channels_(_numChannels), hidden_channels_(_numChannels), filter_size_(_filterSize) {}

		TORCH_ARG(int64_t, num_channels);
		TORCH_ARG(int64_t, hidden_channels);
		TORCH_ARG(int64_t, filter_size);
		TORCH_ARG(int64_t, num_layers) = 1;
		TORCH_ARG(bool, bias) = false;
		TORCH_ARG(ActivationFn, activation) = torch::tanh;
		TORCH_ARG(bool, residual) = false;
		TORCH_ARG(bool, ext_residual) = false;
		TORCH_ARG(bool, symmetric) = false;
	};

	struct ConvolutionalImpl : public torch::nn::Cloneable<ConvolutionalImpl>
	{
		using Options = ConvolutionalOptions;
		explicit ConvolutionalImpl(const Options& _options);

		void reset() override;

		torch::Tensor forward(torch::Tensor _input);

		Options options;
		using Conv = ExtConv1d; // torch::nn::Conv1d
		std::vector<Conv> layers;
		torch::nn::Conv1d residual;
	};

	TORCH_MODULE(Convolutional);

	// specialization for HeatEq with coefficients, only forwards the first channel
	template<>
	struct IdentityMap<Convolutional>
	{
		static torch::Tensor forward(const torch::Tensor& _input)
		{
			using namespace torch::indexing;
			return _input.index({ Slice(), 0, Slice() });;
		}
	};

	enum struct FlatConvMode
	{
		ConstDiffusion,
		ConstTemp,
		Stack // temperature and diffusion are expected a single dimension
	};

	// this wrapper makes it possible to compute the Jacobian of just the first channel with eval::computeJacobian.
	struct FlatConvWrapperImpl : public torch::nn::Module
	{
		explicit FlatConvWrapperImpl(Convolutional _net, FlatConvMode _mode = FlatConvMode::ConstDiffusion);

		torch::Tensor forward(torch::Tensor _input);

		Convolutional net;
		FlatConvMode mode;
		torch::Tensor constantInputs; // either diffusion, temperature or none based on mode
	};

	TORCH_MODULE(FlatConvWrapper);

}
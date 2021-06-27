#pragma once

#include <torch/torch.h>

namespace nn {

	template<size_t D>
	struct ExtConvOptions : public torch::nn::ConvOptions<D>
	{
		using torch::nn::ConvOptions<D>::ConvOptions;
	//	explicit ExtConvOptions(const ConvOptions<D>& _options) : ConvOptions(_options) {}

		TORCH_ARG(int64_t, symmetric) = false;
	};

	template<size_t D>
	struct ExtConvImpl : public torch::nn::Cloneable<ExtConvImpl<D>>
	{
		using Conv = std::conditional_t<D == 1, torch::nn::Conv1d, torch::nn::Conv2d>;
		using Options = ExtConvOptions<D>;
		explicit ExtConvImpl(const Options& _options)
			: convolution(nullptr),
			options(_options)
		{
			reset();
		}

		void reset() override
		{
			convolution = Conv(options);
			// do not register full module since convolution->weight is parametrized
			this->register_parameter("bias", convolution->bias, options.bias());

			if (options.symmetric())
			{
				weight = this->register_parameter("weight", torch::empty(convolution->weight.sizes()), true);
				torch::nn::init::kaiming_uniform_(weight, std::sqrt(5));
			}
			else
			{
				weight = this->register_parameter("weight", convolution->weight, true);
			}
		}

		torch::Tensor forward(torch::Tensor x)
		{
			if(options.symmetric())
				convolution->weight = 0.5 * (weight + weight.flip(weight.dim()-1));
			return convolution(x);
		}

		torch::Tensor weight;
		Conv convolution;
		Options options;
	};

	using ExtConv1dImpl = ExtConvImpl<1>;
	TORCH_MODULE(ExtConv1d);
}
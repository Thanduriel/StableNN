#pragma once

#include <torch/torch.h>

namespace nn {

	template<size_t D>
	struct ExtConvImpl : public torch::nn::Cloneable<ExtConvImpl<D>>
	{
		using Conv = std::conditional_t<D == 1, torch::nn::Conv1d, torch::nn::Conv2d>;
		using Options = torch::nn::ConvOptions<D>;
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
			weight = this->register_parameter("weight", torch::empty(convolution->weight.sizes()), true);
			this->register_parameter("bias", convolution->bias, options.bias());

			torch::nn::init::kaiming_uniform_(weight, std::sqrt(5));
		}

		torch::Tensor forward(torch::Tensor x)
		{
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
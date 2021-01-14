#include "convolutional.hpp"

namespace nn {

	ConvolutionalImpl::ConvolutionalImpl(const Options& _options)
		: options(_options)
	{
		reset();
	}

	void ConvolutionalImpl::reset()
	{
		torch::nn::Conv1dOptions convOptions(1, 1, options.filter_size());
		convOptions.padding_mode(torch::kCircular);
		convOptions.padding(options.filter_size() / 2);

		layers.clear();
		layers.reserve(options.num_layers());

		for (int64_t i = 0; i < options.num_layers(); ++i)
		{
			layers.emplace_back(torch::nn::Conv1d(convOptions));
			register_module("layer" + std::to_string(i), layers.back());
		}
	}

	torch::Tensor ConvolutionalImpl::forward(torch::Tensor x)
	{
		x = x.unsqueeze(x.dim()-1);
		if (x.dim() < 3)
			x = x.unsqueeze(0);
		for (auto& layer : layers)
			x = options.activation()(layer(x));

		return x.squeeze();
	}
}
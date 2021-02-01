#include "convolutional.hpp"

namespace nn {

	ConvolutionalImpl::ConvolutionalImpl(const Options& _options)
		: options(_options)
	{
		reset();
	}

	void ConvolutionalImpl::reset()
	{
		torch::nn::Conv1dOptions convOptions(options.num_channels(), options.num_channels(), options.filter_size());
		convOptions.padding_mode(torch::kCircular);
		convOptions.padding(options.filter_size() / 2);

		layers.clear();
		layers.reserve(options.num_layers());

		for (int64_t i = 0; i < options.num_layers()-1; ++i)
		{
			layers.emplace_back(torch::nn::Conv1d(convOptions));
			register_module("layer" + std::to_string(i), layers.back());
		}

		convOptions.out_channels(1);
		layers.emplace_back(torch::nn::Conv1d(convOptions));
		register_module("layer" + std::to_string(options.num_layers()-1), layers.back());
	}

	torch::Tensor ConvolutionalImpl::forward(torch::Tensor x)
	{
		// channel dimension
		if(x.dim() < 3)
			x = x.unsqueeze(x.dim()-1);
		// batch dimension
		if (x.dim() < 3)
			x = x.unsqueeze(0);

		auto& activation = options.activation();
		for(size_t i = 0; i < layers.size()-1; ++i)
			x = activation(layers[i](x));
		x = layers.back()(x);

		return x.squeeze();
	}
}
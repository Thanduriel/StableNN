#include "tcn.hpp"

namespace nn {

	struct TCNResBlockOptions : public TCNOptions
	{
		TCNResBlockOptions(const TCNOptions& _options, int64_t _dilation)
			: TCNOptions(_options), dilation_(_dilation)
		{}

		TORCH_ARG(int64_t, dilation);
	};

	struct ResLayerImpl : public torch::nn::Cloneable<ResLayerImpl>
	{
		ResLayerImpl(const TCNResBlockOptions& _options)
			: residual(nullptr), dropout_layer(nullptr), options(_options)
		{
			reset();
		}

		void reset() override
		{
			// reset to no layer instead of default constructed
			residual = nullptr;
			dropout_layer = nullptr;
			layers.clear();

			const int64_t stack_size = options.block_size();
			const int64_t dilation = options.dilation();
			const int64_t kernel_size = options.kernel_size();
			const int64_t in_channels = options.in_channels();
			const int64_t out_channels = options.out_channels();

			if (options.average() && dilation > 1) 
			{
				const int kernel = options.dilation() * 2 - 1;
				auto convOptions = torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
					.bias(false)
					.padding(kernel / 2)
					.dilation(kernel)
					.padding_mode(torch::kZeros);

				layers.emplace_back(torch::nn::Conv1d(convOptions));
				register_module("average", layers.back());
			}

			for (int i = 0; i < stack_size; ++i) 
			{
				//	const int dilation = 1 << i;
				const int in = in_channels + (out_channels - in_channels) * i / stack_size;
				const int out = in_channels + (out_channels - in_channels) * (i + 1) / stack_size;
				auto convOptions = torch::nn::Conv1dOptions(in, out, kernel_size)
					.bias(options.bias())
					.padding(kernel_size / 2 * dilation)
					.dilation(dilation)
					.padding_mode(torch::kZeros);

				layers.emplace_back(torch::nn::Conv1d(convOptions));
				register_module("layer" + std::to_string(i), layers.back());
			}

			if (in_channels != out_channels) 
			{
				auto convOptions = torch::nn::Conv1dOptions(in_channels, out_channels, 1).bias(false);
				residual = torch::nn::Conv1d(convOptions);
				register_module("residual", residual);
			}

			// dropout layer does not have weights and can be reused
			if (options.dropout() > 0.0) 
			{
				dropout_layer = torch::nn::Dropout(options.dropout());
				register_module("dropout", dropout_layer);
			}
		}

		torch::Tensor forward(torch::Tensor x) 
		{
			auto in = x.clone();

			auto activation = options.activation();
			for (auto& layer : layers)
			{
				x = activation(layer->forward(x));
				if (dropout_layer)
					x = dropout_layer(x);
			}

			return x + (residual ? residual(in) : in);
		}

		std::vector<torch::nn::Conv1d> layers;
		torch::nn::Conv1d residual;
		torch::nn::Dropout dropout_layer;
		TCNResBlockOptions options;
	};

	TORCH_MODULE(ResLayer);

	TCNOptions::TCNOptions(int64_t in_channels, int64_t out_channels, int64_t window_size)
		: in_channels_(in_channels), 
		hidden_channels_(std::max(in_channels / 2, out_channels)),
		out_channels_(out_channels), 
		window_size_(window_size)
	{
	}

	TCNImpl::TCNImpl(const TCNOptions& _options) : options(_options)
	{
		reset();
	}

	void TCNImpl::reset()
	{
		layers = torch::nn::Sequential();
		// reduce number of channels in first block
		auto resOptions = TCNResBlockOptions(options, 1);
		resOptions.out_channels(options.hidden_channels());
		layers->push_back(ResLayer(resOptions));

		resOptions.in_channels(options.hidden_channels());
		for (int i = 0; i < options.residual_blocks() - 1; ++i)
		{
			layers->push_back(ResLayer(resOptions));
		}

		// convolutional stack to reduce the output size
		if (options.reduce_stack())
		{
			/*	int window = options.window_size();
				while (window > 1) {
					auto opts = torch::nn::Conv1dOptions(hidden_channels, hidden_channels, filter_size)
						.stride(2)
						.bias(options.bias())
						.padding(filter_size / 2)
						.padding_mode(torch::kZeros);
					layers->push_back(convlayer(opts, options.activation()));
					window >>= 1;
				}
				auto opts = torch::nn::Conv1dOptions(hidden_channels, options.out_channels(), 1);
				layers->push_back(convlayer(opts, ACT::none));*/
		}
		else
		{ // alternative is a simple fully connected linear layer
			layers->push_back(torch::nn::Flatten());
			layers->push_back(torch::nn::Linear(
				torch::nn::LinearOptions(options.window_size() * options.hidden_channels(), options.out_channels())
				.bias(false)));
		}
		register_module("conv_layers", layers);
	}

	torch::Tensor TCNImpl::forward(torch::Tensor x) 
	{
		x = layers->forward(x);
		return x.squeeze().unsqueeze(0);
	}

} 
#include "tcn.hpp"

namespace nn {

	template<int64_t D>
	struct TCNResBlockOptions : public TCNOptions<D>
	{
		TCNResBlockOptions(const TCNOptions<D>& _options, int64_t _dilation)
			: TCNOptions<D>(_options), dilation_(_dilation)
		{}

		TORCH_ARG(int64_t, dilation);
	};

	template<int64_t D>
	struct ResLayerImpl : public torch::nn::Cloneable<ResLayerImpl<D>>
	{
		static torch::ExpandingArray<D, int64_t> makeExpandingArray(const int64_t& val)
		{
			torch::ExpandingArray<D, int64_t> arr(1);
			arr->front() = val;
			return arr;
		}

		// todo also make AvgPool and Dropout depended on D
		using Conv = std::conditional_t<D == 1, torch::nn::Conv1d, torch::nn::Conv2d>;

		ResLayerImpl(const TCNResBlockOptions<D>& _options)
			: residual(nullptr), 
			dropout_layer(nullptr),
			avg_residual(nullptr),
			options(_options)
		{
			reset();
		}

		void reset() override
		{
			// currently not supported
			assert(D == 1 || !options.average());
			assert(D == 1 || !options.dropout());

			// reset to no layer instead of default constructed
			residual = nullptr;
			dropout_layer = nullptr;
			avg_residual = nullptr;
			layers.clear();

			const int64_t stack_size = options.block_size();
			const int64_t dilation = options.dilation();
			const auto kernel_size = options.kernel_size();
			const int64_t in_channels = options.in_channels();
			const int64_t out_channels = options.out_channels();

			for (int i = 0; i < stack_size; ++i) 
			{
				const int in = in_channels + (out_channels - in_channels) * i / stack_size;
				const int out = in_channels + (out_channels - in_channels) * (i + 1) / stack_size;
				auto convOptions = torch::nn::ConvOptions<D>(in, out, kernel_size)
					.bias(options.bias())
					.padding(makeExpandingArray(kernel_size->front() / 2 * dilation))
					.dilation(makeExpandingArray(dilation))
					.padding_mode(torch::kZeros)
					.stride(makeExpandingArray(options.average() && i == stack_size - 1 ? 2 : 1));

				layers.emplace_back(Conv(convOptions));
				this->register_module("layer" + std::to_string(i), layers.back());
			}

			if (options.residual())
			{
				if (in_channels != out_channels)
				{
					auto convOptions = torch::nn::ConvOptions<D>(in_channels, out_channels, 1).bias(false);
					residual = Conv(convOptions);
					this->register_module("residual", residual);
				}

				if (options.average())
				{
					auto avgOptions = torch::nn::AvgPool1dOptions(2)
						.stride(2)
						.padding(0)
						.count_include_pad(false);

					avg_residual = this->register_module("avg_residual", torch::nn::AvgPool1d(avgOptions));
				}
			}

			// dropout layer does not have weights and can be reused
			if (options.dropout() > 0.0) 
			{
				dropout_layer = torch::nn::Dropout(options.dropout());
				this->register_module("dropout", dropout_layer);
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

			if (avg_residual)
				in = avg_residual(in);
			if (residual)
				in = residual(in);

			return options.residual() ? x + in : x;
		}

		std::vector<Conv> layers;
		Conv residual;
		torch::nn::AvgPool1d avg_residual;
		torch::nn::Dropout dropout_layer;
		TCNResBlockOptions<D> options;
	};


	template<size_t D, typename Derived>
	TCNImpl<D, Derived>::TCNImpl(const TCNOptions<D>& _options) : options(_options)
	{
		reset();
	}

	template<size_t D, typename Derived>
	void TCNImpl<D, Derived>::reset()
	{
		layers = torch::nn::Sequential();
		// change number of channels in first block
		auto resOptions = TCNResBlockOptions<D>(options, 1);
		resOptions.out_channels(options.hidden_channels());
		layers->push_back(ResLayerImpl<D>(resOptions));

		resOptions.in_channels(options.hidden_channels());
		for (int i = 1; i < options.residual_blocks(); ++i)
		{
			if (!options.average())
				resOptions.dilation() <<= 1;
			layers->push_back(ResLayerImpl<D>(resOptions));
		}

		// fully connected linear layer to reach output size
		layers->push_back(torch::nn::Flatten());
		int64_t internal_size = options.window_size() * options.hidden_channels();
		if (options.average()) // divide by 2 for each block
			internal_size >>= options.residual_blocks();
		layers->push_back(torch::nn::Linear(
			torch::nn::LinearOptions(internal_size, options.out_channels())
			.bias(options.bias())));

		this->register_module("layers", layers);
	}

	template<size_t D, typename Derived>
	torch::Tensor TCNImpl<D, Derived>::forward(torch::Tensor x)
	{
		using namespace torch::indexing;
		auto residual = x.index({ "...", options.window_size() - 1 });
		x = layers->forward(x) + residual;

		return x.squeeze().unsqueeze(0);
	}

	// explicit instantiations
	template class TCNImpl<1, TCN1DImpl>;
} 
#include "tcn.hpp"

namespace nn {

	// expanding array with val as first element, 1 otherwise
	template<int64_t D>
	static torch::ExpandingArray<D, int64_t> makeExpandingArray(int64_t val)
	{
		torch::ExpandingArray<D, int64_t> arr(1);
		arr->front() = val;
		return arr;
	}

	template<int64_t D>
	struct ResLayerImpl : public torch::nn::Cloneable<ResLayerImpl<D>>
	{
		using Conv = std::conditional_t<D == 1, torch::nn::Conv1d, torch::nn::Conv2d>;
		using AvgPool = std::conditional_t<D == 1, torch::nn::AvgPool1d, torch::nn::AvgPool2d>;

		ResLayerImpl(const TemporalConvBlockOptions<D>& _options)
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
					.padding(makeExpandingArray<D>(kernel_size->front() / 2 * dilation))
					.dilation(makeExpandingArray<D>(dilation))
					.padding_mode(torch::kZeros)
					.stride(makeExpandingArray<D>(options.average() && i == stack_size - 1 ? 2 : 1));

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
					auto avgOptions = torch::nn::AvgPoolOptions<D>(makeExpandingArray<D>(2))
						.stride(makeExpandingArray<D>(2))
						.padding(0)
						.count_include_pad(false);

					avg_residual = this->register_module("avg_residual", AvgPool(avgOptions));
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
		AvgPool avg_residual;
		torch::nn::Dropout dropout_layer; // todo: also consider Dropout2D ?
		TemporalConvBlockOptions<D> options;
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
		auto resOptions = TemporalConvBlockOptions<D>(options.in_size()->front(), 
			options.hidden_channels(),
			options.kernel_size())
			.block_size(options.block_size())
			.bias(options.bias())
			.activation(options.activation())
			.dropout(options.dropout())
			.average(options.average())
			.residual(options.residual());
		layers->push_back(ResLayerImpl<D>(resOptions));

		resOptions.in_channels(options.hidden_channels());
		for (int i = 1; i < options.residual_blocks(); ++i)
		{
			if (!options.average())
				resOptions.dilation() <<= 1;
			layers->push_back(ResLayerImpl<D>(resOptions));
		}

		// fully connected linear layer to reach output size
	//	layers->push_back(torch::nn::Flatten());
		int64_t window_size = options.in_size()->at(1);
		if (options.average()) // divide by 2 for each block
			window_size >>= options.residual_blocks();

		using Conv = std::conditional_t<D == 1, torch::nn::Conv1d, torch::nn::Conv2d>;
		auto convOptions = torch::nn::ConvOptions<D>(
			options.hidden_channels(), 
			options.out_size()->front(), 
			makeExpandingArray<D>(window_size))
			.bias(options.bias());
		layers->push_back(Conv(convOptions));
	/*	layers->push_back(torch::nn::Linear(
			torch::nn::LinearOptions(internal_size, options.out_channels())
			.bias(options.bias())));*/
	/*	if constexpr (D > 1)
		{
			layers->push_back(torch::nn::Unflatten(torch::nn::UnflattenOptions(1, {});
		}*/

		this->register_module("layers", layers);
	}

	template<size_t D, typename Derived>
	torch::Tensor TCNImpl<D, Derived>::forward(torch::Tensor x)
	{
		return layers->forward(x).squeeze(2);
	}

	// explicit instantiations
	template class TCNImpl<1, TCN1DImpl>;
	template class TCNImpl<2, TCN2DImpl>;

	torch::Tensor SimpleTCNImpl::forward(torch::Tensor x)
	{
		using namespace torch::indexing;
		auto residual = x.index({ "...", options.in_size()->at(1) - 1 });
		x = TCNImpl<1, SimpleTCNImpl>::forward(x) + residual;

		return x;
	//	return x.squeeze().unsqueeze(0);
	}
	template class TCNImpl<1, SimpleTCNImpl>;
} 
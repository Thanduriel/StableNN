#include "tcn.hpp"

namespace nn {

	// expanding array with val at position ind element, oth otherwise
	template<int64_t D>
	static torch::ExpandingArray<D, int64_t> makeExpandingArray(int64_t val, int64_t oth = 1, int64_t ind = 0)
	{
		torch::ExpandingArray<D, int64_t> arr(oth);
		arr->at(ind) = val;
		return arr;
	}

	template<int64_t D>
	TemporalConvBlockImpl<D>::TemporalConvBlockImpl(const TemporalConvBlockOptions<D>& _options)
		: residual(nullptr),
		avg_residual(nullptr),
		dropout_layer(nullptr),
		options(_options)
	{
		reset();
	}

	template<int64_t D>
	void TemporalConvBlockImpl<D>::reset()
	{
		// currently not supported
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
		// set padding so that the size is not changed by the convolutions
		auto padding = kernel_size;
		assert(kernel_size->front() % 2 == 1 || options.average());
		padding->front() = options.average() ? kernel_size->front() / 2 - 1 + kernel_size->front() % 2
			: (kernel_size->front() - 1) * dilation / 2;
		for (size_t i = 1; i < D; ++i)
		{
			padding->at(i) = kernel_size->at(i) / 2;
			assert(kernel_size->at(i) % 2 == 1);
		}

		for (int i = 0; i < stack_size; ++i)
		{
			const int in = in_channels + (out_channels - in_channels) * i / stack_size;
			const int out = in_channels + (out_channels - in_channels) * (i + 1) / stack_size;
			auto convOptions = torch::nn::ConvOptions<D>(in, out, kernel_size)
				.bias(options.bias())
				.padding(padding)
				.dilation(makeExpandingArray<D>(dilation))
				.padding_mode(options.padding_mode()->front())
				.stride(makeExpandingArray<D>(options.average() && i == stack_size - 1 ? 2 : 1));

			if (options.interleaved())
			{
				// +1 to start with spatial dim
				const int dim = (i+1) % D;
				convOptions.kernel_size(makeExpandingArray<D>(kernel_size->at(dim), 1, dim));
				convOptions.padding(makeExpandingArray<D>(padding->at(dim), 0, dim));
				convOptions.padding_mode(options.padding_mode()->at(dim));
			}

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

	template<int64_t D>
	torch::Tensor TemporalConvBlockImpl<D>::forward(torch::Tensor x)
	{
		torch::Tensor in = options.residual() ? x.clone() : torch::Tensor();

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

	// explicit instantiations
	template class TemporalConvBlockImpl<1>;
	template class TemporalConvBlockImpl<2>;

	// ============================================================================

	template<size_t D>
	TemporalConvBlockOptions<D> makeBlockOptions(int64_t in_channels, int64_t out_channels, const TCNOptions<D>& options)
	{
		return TemporalConvBlockOptions<D>(in_channels,
			out_channels,
			options.kernel_size())
			.block_size(options.block_size())
			.bias(options.bias())
			.activation(options.activation())
			.dropout(options.dropout())
			.average(options.average())
			.residual(options.residual())
			.interleaved(options.interleaved());
	}

	template<size_t D>
	TCNImpl<D>::TCNImpl(const TCNOptions<D>& _options) : options(_options)
	{
		reset();
	}

	template<size_t D>
	void TCNImpl<D>::reset()
	{
		layers = torch::nn::Sequential();
		// change number of channels in first block
		auto resOptions = makeBlockOptions<D>(options.in_size()->front(),
			options.hidden_channels(),
			options);
		layers->push_back(TCNBlock(resOptions));

		resOptions.in_channels(options.hidden_channels());
		for (int i = 1; i < options.residual_blocks(); ++i)
		{
			if (!options.average())
				resOptions.dilation() <<= 1;
			layers->push_back(TCNBlock(resOptions));
		}

		// convolution over the complete remaining time dimension
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

		this->register_module("layers", layers);
	}

	template<size_t D>
	torch::Tensor TCNImpl<D>::forward(torch::Tensor x)
	{
		// remove time and channel dimension if 1
		return layers->forward(x).squeeze(2).squeeze(1);
	}

	// explicit instantiations
	template class TCNImpl<1>;
	template class TCNImpl<2>;

	// ============================================================================

	SimpleTCNImpl::SimpleTCNImpl(const Options& _options)
		: layers(nullptr),
		options(_options)
	{
		reset();
	}

	void SimpleTCNImpl::reset()
	{
		layers = register_module("tcn", TCN1d(options));
	}

	torch::Tensor SimpleTCNImpl::forward(torch::Tensor x)
	{
		using namespace torch::indexing;
		auto residual = x.index({ "...", options.in_size()->at(1) - 1 });
		x = layers->forward(x) + residual;

		return x;
	//	return x.squeeze().unsqueeze(0);
	}

	// ============================================================================

	ExtTCNImpl::ExtTCNImpl(const Options& _options)
		: layers(nullptr),
		options(_options)
	{
		reset();
	}

	void ExtTCNImpl::reset()
	{
		layers = register_module("tcn", TCN2d(options));
	}

	torch::Tensor ExtTCNImpl::forward(torch::Tensor x)
	{
		using namespace torch::indexing;
		/* batch size, channels, time, spatial */
		auto residual = x.index({Slice(), 0, -1, Slice() });
		x = layers->forward(x) + residual;
		return x;
	}
} 
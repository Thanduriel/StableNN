#include "convolutional.hpp"
#include "utils.hpp"

namespace nn {

	ConvolutionalImpl::ConvolutionalImpl(const Options& _options)
		: options(_options),
		residual(nullptr)
	{
		reset();
	}

	void ConvolutionalImpl::reset()
	{
		layers.clear();
		layers.reserve(options.num_layers());
		residual = nullptr;

		nn::ExtConvOptions<1> convOptions(options.num_channels(), options.hidden_channels(), options.filter_size());
		convOptions.padding_mode(torch::kCircular);
		convOptions.padding(options.filter_size() / 2);
		convOptions.bias(options.bias());
		convOptions.symmetric(options.symmetric());

		if (options.num_layers() > 1)
		{
			layers.emplace_back(convOptions);
			register_module("layer" + std::to_string(0), layers.back());
			if (options.residual() && options.num_channels() != options.hidden_channels())
			{
				torch::nn::Conv1dOptions resOptions(options.num_channels(), options.hidden_channels(), 1);
				resOptions.bias(false);
				residual = torch::nn::Conv1d(resOptions);
				register_module("residual", residual);
			}

			convOptions.in_channels(options.hidden_channels());
			for (int64_t i = 1; i < options.num_layers() - 1; ++i)
			{
				layers.emplace_back(convOptions);
				register_module("layer" + std::to_string(i), layers.back());
			}
		}

		// final layer only reduces the number of channels
		convOptions.out_channels(1);
		convOptions.kernel_size(1);
		convOptions.padding(0);
		convOptions.symmetric(false);
		layers.emplace_back(convOptions);
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

		torch::Tensor extResidual;
		if (options.ext_residual()) extResidual = IdentityMap<Convolutional>::forward(x);

		auto& activation = options.activation();

		if (layers.size() > 1)
		{
			torch::Tensor y = activation(layers[0](x));
			if (residual)
				x = residual(x) + y;
			else if (options.residual())
				x = x + y;
			else
				x = y;
			for (size_t i = 1; i < layers.size() - 1; ++i)
			{
				y = activation(layers[i](x));
				x = options.residual() ? x + y : y;
			}
		}
		x = layers.back()(x).squeeze();
		if (extResidual.defined())
			x += extResidual.squeeze();

		return x;
	}

	// ******************************************************************** //
	FlatConvWrapperImpl::FlatConvWrapperImpl(Convolutional _net)
		: net(_net)
	{
	//	net = register_module("net", Convolutional(_options));
	}

	torch::Tensor FlatConvWrapperImpl::forward(torch::Tensor _input)
	{
		torch::Tensor constants = constantInputs.repeat({ _input.size(0), 1 });
		torch::Tensor tensor = torch::cat({ _input.unsqueeze(1), constants.unsqueeze(1) }, 1);
		return net->forward(tensor);
	}
}
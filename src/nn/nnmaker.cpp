#include "nnmaker.hpp"

#include "mlp.hpp"
#include "antisymmetric.hpp"
#include "hamiltonian.hpp"
#include "convolutional.hpp"
#include "tcn.hpp"

namespace nn {

	template<>
	MLPOptions makeOptions<MLPOptions>(const HyperParams& _params)
	{
		return nn::MLPOptions(*_params.get<size_t>("num_inputs_net"))
			.hidden_layers(*_params.get<int>("depth"))
			.hidden_size(*_params.get<int>("hidden_size"))
			.bias(*_params.get<bool>("bias"))
			.activation(*_params.get<nn::ActivationFn>("activation"))
			.total_time(*_params.get<double>("time"));
	}

	template<>
	AntiSymmetricOptions makeOptions<AntiSymmetricOptions>(const HyperParams& _params)
	{
		return AntiSymmetricOptions(*_params.get<size_t>("num_inputs_net"))
			.num_layers(*_params.get<int>("depth"))
			.diffusion(*_params.get<double>("diffusion"))
			.total_time(*_params.get<double>("time"))
			.bias(*_params.get<bool>("bias"))
			.activation(*_params.get<nn::ActivationFn>("activation"));
	}

	template<>
	HamiltonianOptions makeOptions<HamiltonianOptions>(const HyperParams& _params)
	{
		return HamiltonianOptions(*_params.get<size_t>("num_inputs_net"))
			.num_layers(*_params.get<int>("depth"))
			.total_time(*_params.get<double>("time"))
			.bias(*_params.get<bool>("bias"))
			.activation(*_params.get<nn::ActivationFn>("activation"))
			.augment_size(*_params.get<int>("augment"))
			.symmetric(*_params.get<bool>("symmetric"));
	}

	template<>
	ConvolutionalOptions makeOptions<ConvolutionalOptions>(const HyperParams& _params)
	{
		return ConvolutionalOptions(*_params.get<int>("num_channels"), *_params.get<int>("kernel_size"))
			.num_layers(*_params.get<int>("depth"))
			.bias(*_params.get<bool>("bias"))
			.hidden_channels(*_params.get<int>("hidden_channels"))
			.residual(_params.get<bool>("residual", false))
			.activation(*_params.get<nn::ActivationFn>("activation"))
			.symmetric(*_params.get<bool>("symmetric"))
			.ext_residual(*_params.get<bool>("ext_residual"));
	}

	template<>
	TCNOptions<1> makeOptions<TCNOptions<1>>(const HyperParams& _params)
	{
		const int64_t channels = *_params.get<size_t>("num_channels");
		const int64_t inputs = *_params.get<size_t>("num_inputs_net");

		return TCNOptions<1>({ channels, inputs / channels }, { channels }, *_params.get<int>("kernel_size"))
			.hidden_channels(_params.get<int>("hidden_size", channels))
			.residual_blocks(*_params.get<int>("residual_blocks"))
			.residual(*_params.get<bool>("residual"))
			.activation(*_params.get<nn::ActivationFn>("activation"))
			.block_size(*_params.get<int>("block_size"))
			.average(*_params.get<bool>("average"))
			.bias(*_params.get<bool>("bias"))
			.padding_mode(*_params.get<torch::nn::detail::conv_padding_mode_t>("padding_mode"));
	}

	template<>
	TCNOptions<2> makeOptions<TCNOptions<2>>(const HyperParams& _params)
	{
		const int64_t channels = *_params.get<size_t>("num_channels");
		const int64_t inputs = *_params.get<size_t>("num_inputs_net");
		// spatial in out size is currently ignored
		return TCNOptions<2>({ channels, inputs / channels, 1 }, { 1, 1 }, { *_params.get<int>("kernel_size_temp"), *_params.get<int>("kernel_size") })
			.hidden_channels(_params.get<int>("hidden_channels", channels))
			.residual_blocks(*_params.get<int>("residual_blocks"))
			.residual(*_params.get<bool>("residual"))
			.activation(*_params.get<nn::ActivationFn>("activation"))
			.block_size(*_params.get<int>("block_size"))
			.average(*_params.get<bool>("average"))
			.causal(*_params.get<bool>("causal"))
			.bias(*_params.get<bool>("bias"))
			.padding_mode({ *_params.get<torch::nn::detail::conv_padding_mode_t>("padding_mode_temp") , *_params.get<torch::nn::detail::conv_padding_mode_t>("padding_mode")})
			.interleaved(*_params.get<bool>("interleaved"));
	}

	template<>
	ExtTCNOptions makeOptions<ExtTCNOptions>(const HyperParams& _params)
	{
		return ExtTCNOptions(makeOptions<TCNOptions<2>>(_params))
			.ext_residual(*_params.get<bool>("ext_residual"));
	}
}
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
		return nn::MLPOptions(*_params.get<size_t>("num_inputs"))
			.hidden_layers(*_params.get<int>("depth"))
			.hidden_size(*_params.get<int>("hidden_size"))
			.bias(*_params.get<bool>("bias"))
			.activation(*_params.get<nn::ActivationFn>("activation"));
	}

	template<>
	AntiSymmetricOptions makeOptions<AntiSymmetricOptions>(const HyperParams& _params)
	{
		return AntiSymmetricOptions(*_params.get<size_t>("num_inputs"))
			.num_layers(*_params.get<int>("depth"))
			.diffusion(*_params.get<double>("diffusion"))
			.total_time(*_params.get<double>("time"))
			.bias(*_params.get<bool>("bias"))
			.activation(*_params.get<nn::ActivationFn>("activation"));
	}

	template<>
	HamiltonianOptions makeOptions<HamiltonianOptions>(const HyperParams& _params)
	{
		return HamiltonianOptions(*_params.get<size_t>("num_inputs"))
			.num_layers(*_params.get<int>("depth"))
			.total_time(*_params.get<double>("time"))
			.bias(*_params.get<bool>("bias"))
			.activation(*_params.get<nn::ActivationFn>("activation"))
			.augment_size(*_params.get<int>("augment"));
	}

	template<>
	ConvolutionalOptions makeOptions<ConvolutionalOptions>(const HyperParams& _params)
	{
		return ConvolutionalOptions(*_params.get<int>("num_channels"), *_params.get<int>("kernel_size"))
			.num_layers(*_params.get<int>("depth"))
			.bias(*_params.get<bool>("bias"))
			.hidden_channels(*_params.get<int>("hidden_channels"))
			.residual(_params.get<bool>("residual", false))
			.activation(*_params.get<nn::ActivationFn>("activation"));
	}

	template<>
	TCNOptions makeOptions<TCNOptions>(const HyperParams& _params)
	{
		const size_t channels = *_params.get<size_t>("num_channels");
		const size_t inputs = *_params.get<size_t>("num_inputs");
		return TCNOptions(channels, channels, inputs / channels)
			.hidden_channels(_params.get<int>("hidden_size", channels))
			.bias(*_params.get<bool>("bias"))
			.kernel_size(*_params.get<int>("kernel_size"))
			.residual_blocks(*_params.get<int>("residual_blocks"))
			.activation(*_params.get<nn::ActivationFn>("activation"))
			.block_size(*_params.get<int>("block_size"))
			.average(*_params.get<bool>("average"));
	}
}
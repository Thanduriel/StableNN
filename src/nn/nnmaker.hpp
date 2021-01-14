#pragma once

#include "hyperparam.hpp"
#include "inoutwrapper.hpp"

namespace nn {

	template<typename Options>
	Options makeOptions(const HyperParams& _params);

	template<typename Net, bool UseWrapper, int StateSize>
	auto makeNetwork(const nn::HyperParams& _params)
	{
		const size_t numInputsNet = *_params.get<size_t>("num_inputs") * StateSize;
		const size_t numOutputsNet = *_params.get<size_t>("num_outputs") * StateSize;

		// convert params from states to values
		HyperParams params;
		params = _params;
		params["num_inputs"] = numInputsNet;
		params["num_outputs"] = numOutputsNet;

		auto options = makeOptions< typename Net::Impl::Options >(params);
		if constexpr (UseWrapper)
		{
			const size_t hiddenSize = *_params.get<int>("hidden_size");

			options.input_size() = hiddenSize;
			nn::InOutWrapper<Net> net(
				nn::InOutWrapperOptions(numInputsNet, hiddenSize, numOutputsNet)
				.proj_mask(nn::InOutWrapperOptions::ProjectionMask::Id)
				.train_out(*_params.get<bool>("train_out"))
				.train_in(*_params.get<bool>("train_in"))
				.in_out_bias(*_params.get<bool>("in_out_bias"))
				, options);
			net->to(torch::kDouble);
			return net;
		}
		else
		{
			Net net(options);
			net->to(torch::kDouble);
			return net;
		}
	}
}
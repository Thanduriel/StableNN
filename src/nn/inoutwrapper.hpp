#pragma once

#include <torch/torch.h>

namespace nn {

	struct InOutWrapperOptions
	{
		InOutWrapperOptions(int64_t _inputSize, int64_t _hiddenSize, int64_t _outputSize)
			: input_size_(_inputSize), output_size_(_outputSize), hidden_size_(_hiddenSize) {}

		TORCH_ARG(int64_t, input_size);
		TORCH_ARG(int64_t, output_size);
		TORCH_ARG(int64_t, hidden_size);
		TORCH_ARG(bool, train_out) = false;
	};

	template<typename HiddenNet>
	class InOutWrapperImpl : public torch::nn::Cloneable<InOutWrapperImpl<HiddenNet>>
	{
	public:
		template<typename NetOptions>
		InOutWrapperImpl(const InOutWrapperOptions& _options, const NetOptions& _netOptions)
			: options(_options),
			hiddenNet(HiddenNet(_netOptions)),
			inputLayer(nullptr),
			outputLayer(nullptr)
		{
			reset();
		}

		void reset() override
		{
			hiddenNet = this->register_module("hidden", HiddenNet(hiddenNet->options));
			if (options.train_out())
			{
				outputLayer = torch::nn::Linear(torch::nn::LinearOptions(
					options.hidden_size(),
					options.output_size()).bias(false));
				this->register_module("out", outputLayer);
			}
			projection = this->register_parameter("projection",
				torch::eye(options.input_size(), options.hidden_size(), c10::TensorOptions(c10::kDouble)), false);
		}

		torch::Tensor forward(torch::Tensor x)
		{
			if(input_size != hidden_size)
				x = x.matmul(projection);
			x = hiddenNet->forward(x);
			return options.trained_out() ? outputLayer(x) : x.matmul(projection.t());
		}

		InOutWrapperOptions options;
	private:

		torch::Tensor projection;
		torch::nn::Linear inputLayer;
		HiddenNet hiddenNet;
		torch::nn::Linear outputLayer;
	};

	template<typename HiddenNet>
	class InOutWrapper : public torch::nn::ModuleHolder<InOutWrapperImpl<HiddenNet>>
	{
	public:
		using torch::nn::ModuleHolder<InOutWrapperImpl<HiddenNet>>::ModuleHolder;
		using Impl = InOutWrapperImpl<HiddenNet>;
	};
}
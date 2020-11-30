#pragma once

#include <torch/torch.h>

namespace nn {

	struct InOutWrapperOptions
	{
		InOutWrapperOptions(int64_t _inputSize, int64_t _hiddenSize, int64_t _outputSize)
			: input_size_(_inputSize), output_size_(_outputSize), hidden_size_(_hiddenSize) {}

		enum struct ProjectionMask {
			Id,
			IdInterleafed,
			Zero,
		};

		TORCH_ARG(int64_t, input_size);
		TORCH_ARG(int64_t, output_size);
		TORCH_ARG(int64_t, hidden_size);
		TORCH_ARG(bool, train_out) = false; // use projection mask or a trained layer
		TORCH_ARG(ProjectionMask, proj_mask) = ProjectionMask::IdInterleafed;
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
			using namespace torch::indexing;

			hiddenNet = this->register_module("hidden", HiddenNet(hiddenNet->options));
			if (options.train_out())
			{
				outputLayer = torch::nn::Linear(torch::nn::LinearOptions(
					options.hidden_size(),
					options.output_size()).bias(false));
				this->register_module("out", outputLayer);
			}

			const int64_t ratio = options.hidden_size() / options.output_size();
			const double val = 1.0 / std::sqrt(ratio);
			torch::Tensor p;
			using ProjMask = InOutWrapperOptions::ProjectionMask;
			switch(options.proj_mask())
			{
			case ProjMask::IdInterleafed:
				p = torch::eye(options.output_size(), c10::TensorOptions(c10::kDouble));
				p = (p * val).repeat({ 1, ratio });
				break;
			case ProjMask::Id:
			{
				const int64_t halfSizeX = options.output_size() / 2;
				const int64_t halfSizeY = options.hidden_size() / 2;
				p = torch::zeros({ options.output_size(), options.hidden_size() }, c10::TensorOptions(c10::kDouble));
				p.index_put_({ Slice(0,halfSizeX), Slice(0,halfSizeY) }, val);
				p.index_put_({ Slice(halfSizeX), Slice(halfSizeY) }, val);
				break;
			}
			case ProjMask::Zero:
				p = torch::eye(options.input_size(), options.hidden_size(), c10::TensorOptions(c10::kDouble));
				break;
			};
			projection = this->register_parameter("projection", p, false);
		}

		torch::Tensor forward(torch::Tensor x)
		{
			if(options.input_size() != options.hidden_size())
				x = x.matmul(projection);
			x = hiddenNet->forward(x);
			return options.train_out() ? outputLayer(x) : x.matmul(projection.t());
		}

		InOutWrapperOptions options;
		torch::Tensor projection;
		torch::nn::Linear inputLayer;
		HiddenNet hiddenNet;
		torch::nn::Linear outputLayer;
	private:
	};

	template<typename HiddenNet>
	class InOutWrapper : public torch::nn::ModuleHolder<InOutWrapperImpl<HiddenNet>>
	{
	public:
		using torch::nn::ModuleHolder<InOutWrapperImpl<HiddenNet>>::ModuleHolder;
		using Impl = InOutWrapperImpl<HiddenNet>;
	};
}
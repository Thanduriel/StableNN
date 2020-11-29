#pragma once

#include <torch/torch.h>
#include "activation.hpp"

namespace nn {

	class AntiSymmetricCellImpl : public torch::nn::Cloneable<AntiSymmetricCellImpl>
	{
	public:
		AntiSymmetricCellImpl(int64_t _size, double _gamma, bool _bias = true);

		void reset() override;
		void reset_parameters();

		void pretty_print(std::ostream& stream) const override;

		torch::Tensor forward(const torch::Tensor& input);
		torch::Tensor system_matrix() const;

		torch::Tensor weight;
		torch::Tensor diffusion;
		torch::Tensor bias;
		int64_t size;
		double gamma;
		bool useBias;
	};

	TORCH_MODULE(AntiSymmetricCell);

	struct AntiSymmetricOptions
	{
		AntiSymmetricOptions(int64_t _inputSize) : input_size_(_inputSize) {}

		TORCH_ARG(int64_t, input_size);
		TORCH_ARG(int64_t, num_layers) = 1;
		TORCH_ARG(bool, bias) = true;
		TORCH_ARG(double, diffusion) = 1;
		TORCH_ARG(double, total_time) = 1.0;
		TORCH_ARG(ActivationFn, activation) = torch::tanh;
	};

	struct AntiSymmetricImpl : torch::nn::Cloneable<AntiSymmetricImpl>
	{
		using Options = AntiSymmetricOptions;
		AntiSymmetricImpl(int64_t _inputs = 2) : AntiSymmetricImpl(AntiSymmetricOptions(_inputs)) {}
		explicit AntiSymmetricImpl(const AntiSymmetricOptions& _options);

		void reset() override;

		torch::Tensor forward(torch::Tensor x);

		AntiSymmetricOptions options;

	private:
		double timeStep;
		std::vector<AntiSymmetricCell> layers;
	};

	TORCH_MODULE(AntiSymmetric);

}
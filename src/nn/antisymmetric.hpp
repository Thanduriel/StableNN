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

	struct AntiSymmetricImpl : torch::nn::Module
	{
		AntiSymmetricImpl(int64_t _inputs = 2,
			int64_t _hiddenLayers = 1,
			double _diffusion = 0.0,
			double _totalTime = 1.0,
			bool _useBias = false,
			ActivationFn _activation = torch::tanh);

		torch::Tensor forward(torch::Tensor x);

		double timeStep;
		std::vector<AntiSymmetricCell> layers;
		ActivationFn activation;
	};

	using AntiSymmetric = AntiSymmetricImpl;

}
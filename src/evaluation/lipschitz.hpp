#pragma once

#include <vector>
#include <torch/torch.h>

namespace eval {

	namespace details {
		template<typename Layer>
		torch::Tensor getSystemMatrix(const Layer& _layer)
		{
			return _layer.system_matrix();
		}

		template<>
		torch::Tensor getSystemMatrix<torch::nn::Linear>(const torch::nn::Linear& _layer)
		{
			return _layer->weight;
		}
	}

	// from "Parseval Networks: Improving Robustness to Adversarial Examples"
	// expects ResNet using activations with lipschitz constant <= 1
	template<typename Layer>
	double lipschitzParseval(const std::vector<Layer>& _layers)
	{
		double p = 1.0;
		for (const Layer& layer : _layers)
		{
			p = p + p * torch::linalg_norm(details::getSystemMatrix(layer), 2).item<double>();
		}

		return p;
	}

	// "Spectrally-normalized margin bounds for neural networks"
	template<typename Layer>
	double spectralComplexity(const std::vector<Layer>& _layers)
	{
		// spectral norms can be reused
		std::vector<double> spectralNorms;
		for (const Layer& layer : _layers)
			spectralNorms.push_back(torch::linalg_norm(details::getSystemMatrix(layer), 2).item<double>());

		double p1 = 1.0;
		for (double s : spectralNorms) p1 *= s;

		double p2 = 0.0;
		for (size_t i = 0; i < _layers.size(); ++i)
		{
			const auto A = details::getSystemMatrix(_layers[i]);
			const auto I = torch::eye(A.size(0), A.size(1));
			// todo use ||...||_2,1 norm instead
			p2 += std::pow(torch::linalg_norm(A - I, 1).item<double>() / spectralNorms[i], 2.0 / 3.0);
		}

		return p1 * std::pow(p2, 3.0 / 2.0);
	}
}
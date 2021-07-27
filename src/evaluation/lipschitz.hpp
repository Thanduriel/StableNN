#pragma once

#include "../nn/inoutwrapper.hpp"
#include <vector>
#include <torch/torch.h>

namespace nn {
	class MultiLayerPerceptron;
	class HamiltonianInterleaved;
	class AntiSymmetric;
}

namespace eval {

	namespace details {
		template<typename Layer>
		torch::Tensor getSystemMatrix(const Layer& _layer)
		{
			return _layer.systemMatrix();
		}

		template<>
		inline torch::Tensor getSystemMatrix<torch::nn::Linear>(const torch::nn::Linear& _layer)
		{
			return _layer->weight;
		}

		inline double norm(const torch::Tensor& _tensor, torch::Scalar _ord)
		{
			const torch::Tensor n = torch::linalg_norm(_tensor, _ord);
			return n.item<double>();
		}
	}

	double lipschitz(const nn::MultiLayerPerceptron& _net);
	double lipschitz(const nn::HamiltonianInterleaved& _net);
	double lipschitz(const nn::AntiSymmetric& _net);

	template<typename HiddenNet>
	double lipschitz(const nn::InOutWrapper<HiddenNet>& _net)
	{
		double p = _net->options.train_in() ? details::norm(_net->inputLayer->weight, 2) : 1.0;
		
		p *= lipschitz(_net->hiddenNet);

		return p * (_net->options.train_out() ? details::norm(_net->outputLayer->weight, 2) : 1.0);
	}

	// from "Parseval Networks: Improving Robustness to Adversarial Examples"
	// expects ResNet using activations with lipschitz constant <= 1
	// this is the same as the lipschitz() routines from above
	template<typename Layer>
	double lipschitzParseval(const std::vector<Layer>& _layers)
	{
		double p = 1.0;
		for (const Layer& layer : _layers)
		{
			torch::Tensor n = torch::linalg_norm(details::getSystemMatrix(layer), 2);
			p = p + p * n.item<double>();
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
		{
			torch::Tensor n = torch::linalg_norm(details::getSystemMatrix(layer), 2);
			spectralNorms.push_back(n.item<double>());
		}

		double p1 = 1.0;
		for (double s : spectralNorms) p1 *= s;

		double p2 = 0.0;
		for (size_t i = 0; i < _layers.size(); ++i)
		{
			const auto A = details::getSystemMatrix(_layers[i]);
			const auto I = torch::eye(A.size(0), A.size(1));
			// todo use ||...||_2,1 norm instead
			const torch::Tensor n = torch::linalg_norm(A - I, 1);
			p2 += std::pow(n.item<double>() / spectralNorms[i], 2.0 / 3.0);
		}

		return p1 * std::pow(p2, 3.0 / 2.0);
	}
}
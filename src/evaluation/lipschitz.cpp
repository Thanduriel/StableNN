#include "lipschitz.hpp"
#include "../nn/mlp.hpp"
#include "../nn/hamiltonian.hpp"
#include "../nn/antisymmetric.hpp"


namespace eval {

	double lipschitz(const nn::MultiLayerPerceptron& _net)
	{
		double p = 1.0;
		for (const auto& layer : _net->layers)
		{
			p *= 1.0 + _net->timeStep * details::norm(layer->weight, 2);
		}

		return p;
	}

	double lipschitz(const nn::HamiltonianInterleaved& _net)
	{
		double p = 1.0;
		for (const auto& layer : _net->layers)
		{
			const auto W0 = layer->systemMatrix(true);
			const auto W1 = layer->systemMatrix(false);
			p *= 1.0 + _net->timeStep * details::norm(W0, 2);
			p *= 1.0 + _net->timeStep * details::norm(W1, 2);
		}

		return p;
	}

	double lipschitz(const nn::AntiSymmetric& _net)
	{
		double p = 1.0;
		for (const auto& layer : _net->layers)
		{
			p *= 1.0 + _net->timeStep * details::norm(layer->systemMatrix(), 2);
		}

		return p;
	}
}
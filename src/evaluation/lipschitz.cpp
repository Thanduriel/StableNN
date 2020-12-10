#include "lipschitz.hpp"
#include "../nn/mlp.hpp"
#include "../nn/hamiltonian.hpp"
#include "../nn/antisymmetric.hpp"


namespace eval {

	double lipschitz(const nn::MultiLayerPerceptron& _net)
	{
		double p = 1.0;
		for (const auto& layer : _net->hiddenLayers)
		{
			p *= 1.0 + details::norm(layer->weight, 2);
		}

		return p;
	}

	double lipschitz(const nn::HamiltonianInterleafed& _net)
	{
		double p = 1.0;
		for (const auto& layer : _net->layers)
		{
			p *= 1.0 + _net->timeStep * details::norm(layer->system_matrix(), 2);
		}

		return p;
	}

	double lipschitz(const nn::AntiSymmetric& _net)
	{
		double p = 1.0;
		for (const auto& layer : _net->layers)
		{
			p *= 1.0 + _net->timeStep * details::norm(layer->system_matrix(), 2);
		}

		return p;
	}
}
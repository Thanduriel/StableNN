#include "stability.hpp"

namespace eval {

	template<>
	void checkLayerStability<torch::nn::Linear>(const torch::nn::Linear& _layer)
	{
		const auto& [eigenvalues, _] = torch::eig(_layer->weight);
		std::cout << eigenvalues << "\n";
	}
}
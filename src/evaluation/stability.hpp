#pragma once

#include <torch/torch.h>

namespace eval {

	template<typename LayerT>
	void checkLayerStability(const LayerT& _layer);

	template<typename Module>
	void checkModuleStability(Module& _module)
	{
		torch::Tensor J = computeJacobian(_module, torch::zeros({ 8 }, c10::TensorOptions(c10::kDouble)));
		const auto& [eigenvalues, _] = torch::eig(J);
		std::cout << eigenvalues << "\n";
	}

	template<typename Module>
	torch::Tensor computeJacobian(Module& _module, const torch::Tensor& _inputs)
	{
		const int64_t n = _inputs.size(0);
		torch::Tensor x = _inputs.repeat({ n,1 });
		x.requires_grad_(true);
		torch::Tensor y = _module.forward(x);
		y.backward(torch::eye(n));

		return x.grad();
	}
}
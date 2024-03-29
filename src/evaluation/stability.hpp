#pragma once

#include <torch/torch.h>
#include <complex>

namespace eval {

	template<typename Module>
	void checkModuleStability(Module& _module)
	{
		torch::Tensor J = computeJacobian(_module, torch::zeros({ 8 }, c10::TensorOptions(c10::kDouble)));
		const auto& [eigenvalues, _] = torch::eig(J);
		std::cout << eigenvalues << "\n";
	}

	// Gives the gradient of _module with respect to the inputs.
	// Expects 1D inputs.
	template<typename Module>
	torch::Tensor computeJacobian(Module& _module, const torch::Tensor& _inputs)
	{
		const int64_t n = _inputs.size(0);
		torch::Tensor x = _inputs.squeeze().repeat({ n,1 });
		x.requires_grad_(true);
		torch::Tensor y = _module->forward(x);
		const int64_t m = y.sizes().back();
		y.backward(torch::eye(n,m));

		return x.grad();
	}

	// Compute eigenvalues of a square matrix.
	std::vector<std::complex<double>> computeEigs(const torch::Tensor& _tensor);

	// Returns a matrix A such that A*x is equivalent to the application of _conv.
	torch::Tensor toMatrix(const torch::nn::Conv1d& _conv, int64_t _size);
	torch::Tensor toMatrix(const torch::Tensor& _kernel, int64_t _size);

	// Directly computes the eigenvalues of a circular matrix constructed from _conv like toMatrix.
	torch::Tensor eigs(const torch::nn::Conv1d& _conv, int64_t _size);

	// Checks that this convolution reduces the amount of total energy.
	// @return eigenvalues of A^T A - diag(0,I)
	void checkEnergy(const torch::nn::Conv1d& _conv, int64_t _size);
	std::vector<double> checkEnergy(const torch::Tensor& _netLinear, int64_t _stateSize = 0);
}
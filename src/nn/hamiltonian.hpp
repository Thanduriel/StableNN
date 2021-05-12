#pragma once

#include "activation.hpp"

namespace nn {

	struct HamiltonianOptions
	{
		HamiltonianOptions(int64_t _inputSize) : input_size_(_inputSize) {}

		TORCH_ARG(int64_t, input_size);
		TORCH_ARG(int64_t, num_layers) = 1;
		TORCH_ARG(bool, bias) = true;
		TORCH_ARG(bool, symmetric) = false; // only used in HamiltonianInterleafed
		TORCH_ARG(int64_t, augment_size) = 1; // only used in HamiltonianAugmented
		TORCH_ARG(double, total_time) = 1.0;
		TORCH_ARG(ActivationFn, activation) = torch::tanh;
	};

	// Dense network with second-order discretization scheme. Imposes no restrictions on the weight matrix.
	class HamiltonianImpl : public torch::nn::Cloneable<HamiltonianImpl>
	{
	public:
		using Options = HamiltonianOptions;
		HamiltonianImpl(int64_t _inputs = 2) : HamiltonianImpl(HamiltonianOptions(_inputs)) {}
		explicit HamiltonianImpl(const HamiltonianOptions& _options);

		void reset() override;

		torch::Tensor forward(const torch::Tensor& _input);

		HamiltonianOptions options;
	private:
		double timeStep;
		std::vector<torch::nn::Linear> layers;
	};

	TORCH_MODULE(Hamiltonian);

	// Affine transformation of a two part state.
	class HamiltonianCellImpl : public torch::nn::Cloneable<HamiltonianCellImpl>
	{
	public:
		HamiltonianCellImpl(int64_t _stateSize, int64_t _augmentSize, bool _bias = true, bool _symmetric = false);

		void reset() override;
		void reset_parameters();

		void pretty_print(std::ostream& stream) const override;

		torch::Tensor forwardY(const torch::Tensor& input);
		torch::Tensor forwardZ(const torch::Tensor& input);
		// anti symmetric matrix that represents both steps
		torch::Tensor system_matrix(bool y) const;
		// only the weight matrix, could be parameterized
		torch::Tensor system_matrix() const;

		torch::Tensor weight;
		torch::Tensor biasY;
		torch::Tensor biasZ;
		int64_t size;
		int64_t augmentSize;
		bool useBias;
		bool symmetric = false;
	};

	TORCH_MODULE(HamiltonianCell);

	// Hamiltonian with state augmented by a zero initialized momentum and with leap frog discretization.
	// Weights will be of size input_size x augment_size.
	class HamiltonianAugmentedImpl : public torch::nn::Cloneable<HamiltonianAugmentedImpl>
	{
	public:
		using Options = HamiltonianOptions;
		HamiltonianAugmentedImpl(int64_t _inputs = 2) : HamiltonianAugmentedImpl(HamiltonianOptions(_inputs)){}
		explicit HamiltonianAugmentedImpl(const HamiltonianOptions& _options);

		void reset() override;

		torch::Tensor forward(const torch::Tensor& _input);

		HamiltonianOptions options;
		double timeStep;
		std::vector<HamiltonianCell> layers;
	};

	TORCH_MODULE(HamiltonianAugmented);

	// Layers like HamiltonianAugmented but the actual state is split evenly between position and momentum.
	class HamiltonianInterleafedImpl : public torch::nn::Cloneable<HamiltonianInterleafedImpl>
	{
	public:
		using Options = HamiltonianOptions;
		HamiltonianInterleafedImpl(int64_t _inputs = 2) : HamiltonianInterleafedImpl(HamiltonianOptions(_inputs)) {}
		explicit HamiltonianInterleafedImpl(const HamiltonianOptions& _options);

		void reset() override;

		torch::Tensor forward(const torch::Tensor& _input);

		HamiltonianOptions options;
		double timeStep;
		std::vector<HamiltonianCell> layers;
	};

	TORCH_MODULE(HamiltonianInterleafed);
}
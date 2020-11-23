#pragma once

#include "activation.hpp"

namespace nn {

	struct HamiltonianOptions
	{
		HamiltonianOptions(int64_t _inputSize) : input_size_(_inputSize) {}

		TORCH_ARG(int64_t, input_size);
		TORCH_ARG(int64_t, num_layers) = 1;
		TORCH_ARG(bool, bias) = true;
		TORCH_ARG(int64_t, augment_size) = 1; // only used in HamiltonianAugmented
		TORCH_ARG(double, total_time) = 1.0;
		TORCH_ARG(ActivationFn, activation) = torch::tanh;
	};

	class HamiltonianImpl : public torch::nn::Cloneable<HamiltonianImpl>
	{
	public:
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

	class HamiltonianCellImpl : public torch::nn::Cloneable<HamiltonianCellImpl>
	{
	public:
		HamiltonianCellImpl(int64_t _stateSize, int64_t _augmentSize, bool _bias = true);

		void reset() override;
		void reset_parameters();

		void pretty_print(std::ostream& stream) const override;

		torch::Tensor forwardY(const torch::Tensor& input);
		torch::Tensor forwardZ(const torch::Tensor& input);

		torch::Tensor weight;
		torch::Tensor biasY;
		torch::Tensor biasZ;
		int64_t size;
		int64_t augmentSize;
		bool useBias;
	};

	TORCH_MODULE(HamiltonianCell);

	class HamiltonianAugmentedImpl : public torch::nn::Cloneable<HamiltonianAugmentedImpl>
	{
	public:
		HamiltonianAugmentedImpl(int64_t _inputs = 2) : HamiltonianAugmentedImpl(HamiltonianOptions(_inputs)){}

		explicit HamiltonianAugmentedImpl(const HamiltonianOptions& _options);

		void reset() override;

	//	HamiltonianAugmentedNet(torch::serialize::InputArchive& archive);
	//	void save(torch::serialize::OutputArchive& archive) const override;

		torch::Tensor forward(const torch::Tensor& _input);

		HamiltonianOptions options;
	private:
		double timeStep;
		std::vector<HamiltonianCell> layers;

	};

	//using HamiltonianAugmented = HamiltonianAugmentedImpl;
	TORCH_MODULE(HamiltonianAugmented);
}
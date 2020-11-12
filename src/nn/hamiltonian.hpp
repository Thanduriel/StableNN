#pragma once

#include "activation.hpp"

namespace nn {

	class HamiltonianNet : public torch::nn::Module
	{
	public:
		HamiltonianNet(int64_t _inputs = 2,
			int64_t _hiddenLayers = 1,
			double _totalTime = 1.0,
			bool _useBias = true,
			ActivationFn _activation = torch::tanh);

		torch::Tensor forward(const torch::Tensor& _input);

		double timeStep;
		std::vector<torch::nn::Linear> layers;
		ActivationFn activation;
	};

	class HamiltonianImpl : public torch::nn::Cloneable<HamiltonianImpl>
	{
	public:
		HamiltonianImpl(int64_t _stateSize, int64_t _augmentSize, bool _bias = true);

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

	TORCH_MODULE(Hamiltonian);

	class HamiltonianAugmentedNet : public torch::nn::Module
	{
	public:
		HamiltonianAugmentedNet(
			int64_t _inputs = 2,
			int64_t _hiddenLayers = 32,
			double _totalTime = 1.0,
			bool _useBias = true,
			ActivationFn _activation = torch::tanh,
			int64_t _augmentSize = 2);

	//	HamiltonianAugmentedNet(torch::serialize::InputArchive& archive);
	//	void save(torch::serialize::OutputArchive& archive) const override;

		torch::Tensor forward(const torch::Tensor& _input);

		double timeStep;
		ActivationFn activation;
	private:
		std::vector<Hamiltonian> layers;
		torch::nn::Linear outputLayer;
		int64_t augmentSize;
	};
}
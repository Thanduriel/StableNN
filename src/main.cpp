#include "systems/pendulum.hpp"
#include "discretization.hpp"
#include "nn/mlp.hpp"
#include "nn/dataset.hpp"
#include "nn/nnintegrator.hpp"
#include "evaluation/renderer.hpp"
#include "evaluation/evaluation.hpp"
#include "nn/hyperparam.hpp"
#include "generator.hpp"
#include "nn/antisymmetric.hpp"
#include "nn/hamiltonian.hpp"

#include <type_traits>
#include <torch/torch.h>
#include <chrono>
#include <iostream>
#include <cmath>


constexpr size_t NUM_TIME_STEPS = 4;

template<typename Network>
void evaluate(Network& _network)
{
	using System = systems::Pendulum<double>;
	System pendulum(0.1, 9.81, 0.5);
	const System::State initialState{ -1.5, 0.0 };

	discretization::LeapFrog<System> leapFrog(pendulum, 0.1);
	discretization::ForwardEuler<System> forwardEuler(pendulum, 0.1);

	std::array<System::State, NUM_TIME_STEPS-1> initialStates;
	initialStates[0] = initialState;
	for (size_t i = 1; i < initialStates.size(); ++i)
	{
		initialStates[i] = leapFrog(initialStates[i-1]);
	}
	nn::Integrator<System, Network, 4> neuralNet(_network, initialStates);

	for (int i = 0; i < 1; ++i)
	{
		eval::PendulumRenderer renderer(0.1);
		renderer.addIntegrator([&, state= leapFrog(initialStates.back())]() mutable
			{
				state = neuralNet(state);
				return state.position;
			});
		renderer.run();
	}

	eval::evaluate(pendulum, leapFrog(initialStates.back()), leapFrog, forwardEuler, neuralNet);
}

int main()
{
	auto start = std::chrono::high_resolution_clock::now();
	systems::Pendulum<double> pendulum(0.1, 9.81, 0.5);
	using Integrator = discretization::LeapFrog<systems::Pendulum<double>>;
	using State = systems::Pendulum<double>::State;
	Integrator integrator(pendulum, 0.001);

	State state{ 1.5, 0.0 };
	State state2{ -2.8, 0.2 };
	State state3{ 0.5, -1.01 };

	State validState1{ 1.3, 2.01 };
	State validState2{ -1.5, 0.0 };
	
/*	visual::PendulumRenderer renderer(0.1);
	renderer.addIntegrator([&, state= state3]() mutable
		{
			state = integrator(state);
			return state.position;
		});
	renderer.run();*/

	DataGenerator generator(pendulum, integrator);

	auto dataset = generator.generate({ state, state2, state3 }, 256, 100, 4, false)
		.map(torch::data::transforms::Stack<>());
	auto validationSet = generator.generate({ validState1, validState2 }, 256, 100, 4, false)
		.map(torch::data::transforms::Stack<>());

	nn::AntiSymmetric bestNet;

	auto trainNetwork = [=, &bestNet](const nn::HyperParams& _params)
	{
		auto data_loader = torch::data::make_data_loader(
			dataset,
			torch::data::DataLoaderOptions().batch_size(64));
		auto validationLoader = torch::data::make_data_loader(
			validationSet,
			torch::data::DataLoaderOptions().batch_size(64));

		//nn::MultiLayerPerceptron net(2, 2, 2, 32, true);
		nn::AntiSymmetric net(8,
			_params.get<int>("depth", 32),
			_params.get<double>("diffusion", 0.0),
			_params.get<double>("time", 10.0),
			_params.get<bool>("bias", true),
			_params.get<nn::ActivationFn>("activation", torch::tanh));
		/*nn::HamiltonianAugmented net(nn::HamiltonianOptions(2)
			.num_layers(_params.get<int>("depth", 32))
			.total_time(_params.get<double>("time", 1.0))
			.bias(_params.get<bool>("bias", true))
			.activation(torch::tanh)
			.augment_size(_params.get<int>("augment", 2)));*/
		net.to(torch::kDouble);

		torch::optim::Adam optimizer(net.parameters(), 
			torch::optim::AdamOptions(_params.get<double>("lr", 0.0001))
				.weight_decay(_params.get<double>("weight_decay", 0.0))
				.amsgrad(_params.get<bool>("amsgrad", false)));

		double bestLoss = std::numeric_limits<double>::max();

		//std::ofstream lossFile("loss.txt");

		for (int64_t epoch = 1; epoch <= 512; ++epoch)
		{
			torch::Tensor totalLoss = torch::zeros({ 1 });
			for (torch::data::Example<>& batch : *data_loader)
			{
				torch::Tensor output = net.forward(batch.data);
				torch::Tensor loss = torch::mse_loss(output, batch.target);
				loss.backward();

				totalLoss += loss;

				optimizer.step();
			}
			// validation
			torch::Tensor validLoss = torch::zeros({ 1 });
			for (torch::data::Example<>& batch : *validationLoader)
			{
				torch::Tensor output = net.forward(batch.data);
				torch::Tensor loss = torch::mse_loss(output, batch.target);
				validLoss += loss;
			}

			const double totalLossD = validLoss.item<double>();
			if (totalLossD < bestLoss)
			{
				bestNet = net;
				bestLoss = totalLossD;
				std::cout << totalLossD << "\n";
			}

		//	lossFile << totalLoss.item<double>() << ", " << totalLossD << "\n";
		//	std::cout << "finished epoch with loss: " << totalLoss.item<double>() << "\n";
		}
		if (bestLoss < 2.0)
		{
		//	torch::save(bestNet, _params.get<std::string>("name", "net.pt"));
			torch::serialize::OutputArchive outputArchive;
			net.save(outputArchive);
			outputArchive.save_to(_params.get<std::string>("name", "net.pt"));
		}

		return bestLoss;
	};
	
/*	auto trainTest = [count = 10.0](const nn::HyperParams& param) mutable
	{
	//	std::cout << param.get<double>("decay", 0.0) << std::endl;
		return --count;
	};*/
	/*nn::GridSearchOptimizer hyperOptimizer(trainTest,
		{ {"lr", {1.0e-4, 1.0e-5, 1.0e-6}},
		  {"weight_decay", {0.0, 1.0e-4, 1.0e-5, 1.0e-6}},
		  {"amsgrad", {false, true}} });*/
	nn::GridSearchOptimizer hyperOptimizer(trainNetwork,
		{ {"length", {2, 4, 8, 16, 32, 64}},
		  {"time", {0.1, 1.0, 4.0, 8.0, 16.0}},
		  {"bias", {false, true}},
		  {"diffusion", {0.0, 1.0e-6, 1.0e-5, 1.0e-4}},
		  {"activation", {nn::ActivationFn(torch::tanh), nn::ActivationFn(torch::relu), nn::ActivationFn(torch::sigmoid)}}});

//	hyperOptimizer.run(8);
	nn::HyperParams params;
	params["lr"] = 1e-04;
	params["weight_decay"] = 1e-6;
	params["depth"] = 32;
	params["diffusion"] = 0.0;
	params["bias"] = true;
	params["time"] = 4.0;

	std::cout << trainNetwork(params) << " ";
	const auto& [eigenvalues, _] = torch::eig(bestNet.layers[0]->system_matrix());
	std::cout << eigenvalues;
/*	torch::serialize::InputArchive archive;
	archive.load_from("net.pt");
	nn::AntiSymmetricNet savedNet(2, 2, 2, 2, true);
	savedNet.to(c10::kDouble);
//	torch::load(savedNet, "net.pt");
	savedNet.load(archive);
	evaluate(savedNet);*/
	evaluate(bestNet);
}
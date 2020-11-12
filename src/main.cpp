#include "systems/pendulum.hpp"
#include "discretization.hpp"
#include "nn/mlp.hpp"
#include "nn/dataset.hpp"
#include "nn/nnintegrator.hpp"
#include "visualisation/renderer.hpp"
#include "nn/hyperparam.hpp"
#include "generator.hpp"
#include "nn/antisymmetric.hpp"
#include "nn/hamiltonian.hpp"

#include <type_traits>
#include <torch/torch.h>
#include <chrono>
#include <iostream>

template<typename Network>
void evaluate(Network& _network)
{
	systems::Pendulum<double> pendulum(0.1, 9.81, 0.5);
	systems::Pendulum<double>::State state{ -1.5, 0.0 };
	auto state1 = state;
	auto state2 = state;
	auto state3 = state;

	discretization::LeapFrog<systems::Pendulum<double>> leapFrog(pendulum, 0.1);
	discretization::ForwardEuler<systems::Pendulum<double>> forwardEuler(pendulum, 0.1);
	nn::Integrator<Network, double> neuralNet(_network);

	std::cout << "initial energy: " << pendulum.energy(state) << std::endl;

	for (int i = 0; i < 1; ++i)
	{
		visual::PendulumRenderer renderer(0.1);
		renderer.addIntegrator([&, state]() mutable
			{
				state = neuralNet(state);
				return state.position;
			});
		renderer.run();
	}

	// short term
	for (int i = 0; i < 64; ++i)
	{
		state1 = leapFrog(state1);
		state2 = forwardEuler(state2);
		state3 = neuralNet(state3);
		std::cout << pendulum.energy(state1) << ", " << pendulum.energy(state2) << ", " << pendulum.energy(state3) << "\n";
	}

	// long term energy behaviour
	std::cout << "longterm=======================================" << "\n";
	for (int i = 0; i < 1; ++i)
	{
		for (int j = 0; j < 2048; ++j)
		{
			state1 = leapFrog(state1);
			state2 = forwardEuler(state2);
			state3 = neuralNet(state3);
		}

	//	std::cout << state1.position << "; " << state1.velocity << std::endl;
	//	std::cout << state3.position << ", " << state3.velocity << std::endl;
		std::cout << pendulum.energy(state1) << ", " << pendulum.energy(state2) << ", " << pendulum.energy(state3) << "\n";
	}
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

	auto dataset = generator.generate({ state, state2, state3 }, 256, 100, 1)
		.map(torch::data::transforms::Stack<>());
	auto validationSet = generator.generate({ validState1, validState2 }, 256, 100, 1)
		.map(torch::data::transforms::Stack<>());

	nn::HamiltonianAugmentedNet bestNet;

	auto trainNetwork = [=, &bestNet](const nn::HyperParams& _params)
	{
		auto data_loader = torch::data::make_data_loader(
			dataset,
			torch::data::DataLoaderOptions().batch_size(64));
		auto validationLoader = torch::data::make_data_loader(
			validationSet,
			torch::data::DataLoaderOptions().batch_size(64));

		//nn::MultiLayerPerceptron net(2, 2, 2, 32, true);
		/*nn::AntiSymmetricNet net(2,
			_params.get<int>("depth", 32),
			_params.get<double>("diffusion", 0.0),
			_params.get<double>("time", 10.0),
			_params.get<bool>("bias", true),
			_params.get<nn::ActivationFn>("activation", torch::tanh));*/
		nn::HamiltonianAugmentedNet net(2, 
			_params.get<int>("depth", 32), 
			_params.get<double>("time", 1.0), 
			_params.get<bool>("bias", true), 
			torch::tanh, 
			_params.get<int>("augment", 2));
		//nn::HamiltonianNet net(2, 32, 10.0);
		net.to(torch::kDouble);

		torch::optim::Adam optimizer(net.parameters(), 
			torch::optim::AdamOptions(_params.get<double>("lr", 0.0001))
				.weight_decay(_params.get<double>("weight_decay", 0.0))
				.amsgrad(_params.get<bool>("amsgrad", false)));

		double bestLoss = std::numeric_limits<double>::max();

		std::ofstream lossFile("loss.txt");

		for (int64_t epoch = 1; epoch <= 1; ++epoch)
		{
			torch::Tensor totalLoss = torch::zeros({ 1 });
			for (torch::data::Example<>& batch : *data_loader)
			{
				/*	std::cout << "\n=======\n" << batch.data[0] << "\n" << batch.target[0] << "\n";
					Integrator testInt(pendulum, 0.1);
					auto s = testInt(*reinterpret_cast<State*>(batch.data[0].data<double>()));
					std::cout << s.position << ", " << s.velocity << std::endl;*/
				torch::Tensor output = net.forward(batch.data);
				torch::Tensor loss = torch::mse_loss(output, batch.target);
			//	std::cout << loss.grad_fn()->name();
			//	return 0.0;
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

			lossFile << totalLoss.item<double>() << ", " << totalLossD << "\n";
		//	std::cout << "finished epoch with loss: " << totalLoss.item<double>() << "\n";
		}
	//	if (bestLoss < 2.0)
		{
			torch::serialize::OutputArchive outputArchive;
			bestNet.save(outputArchive);
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
//	params["lr"] = 1e-05;
//	params["weight_decay"] = 1e-6;
/*	params["depth"] = 32;
	params["diffusion"] = 0.1;
	params["bias"] = false;
	params["time"] = 4.0;*/

	//({ std::pair{"lr", 1e-05}, std::pair{"weight_decay", 1e-06} });
	std::cout << trainNetwork(params) << " ";
	torch::serialize::InputArchive archive;
	archive.load_from("net.pt");
	nn::HamiltonianAugmentedNet savedNet;
	savedNet.load(archive);
	evaluate(savedNet);
	evaluate(bestNet);
}
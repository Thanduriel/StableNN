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
#include "evaluation/stability.hpp"

#include <type_traits>
#include <torch/torch.h>
#include <chrono>
#include <iostream>
#include <cmath>


template<size_t NumTimeSteps, typename Network>
void evaluate(Network& _network)
{
	using System = systems::Pendulum<double>;
	System pendulum(0.1, 9.81, 0.5);
	System::State initialState{ -0.5, -0.0 };

	discretization::LeapFrog<System> leapFrog(pendulum, 0.1);
	discretization::ForwardEuler<System> forwardEuler(pendulum, 0.1);

	auto forwardIntegrate = [&](const System::State _state)
	{
		discretization::LeapFrog<System> forward(pendulum, 0.001);
		auto state = _state;
		for (int i = 0; i < 100; ++i)
			state = forward(state);
		return state;
	};

	// prepare initial time series
	std::array<System::State, NumTimeSteps-1> initialStates;
	if (NumTimeSteps > 1)
	{
		initialStates[0] = initialState;
		for (size_t i = 1; i < initialStates.size(); ++i)
		{
			initialStates[i] = forwardIntegrate(initialStates[i - 1]);
		}
		initialState = forwardIntegrate(initialStates.back());
	}
	nn::Integrator<System, Network, NumTimeSteps> neuralNet(_network, initialStates);

	for (int i = 0; i < 1; ++i)
	{
		eval::PendulumRenderer renderer(0.1);
		renderer.addIntegrator([&, state= initialState]() mutable
			{
				state = neuralNet(state);
				return state.position;
			});
		renderer.run();
	}

	eval::evaluate(pendulum, initialState, leapFrog, forwardIntegrate, neuralNet);
}

nn::AntiSymmetric getAntiSymmetric01()
{
	nn::AntiSymmetric net(32,
		64,
		0.001,
		5.0,
		false,
		torch::tanh);

	return net;
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
	State state4{ -0.2, 0.02 };


	State validState1{ 1.3, 2.01 };
	State validState2{ -1.5, 0.0 };
	
/*	visual::PendulumRenderer renderer(0.1);
	renderer.addIntegrator([&, state= state3]() mutable
		{
			state = integrator(state);
			return state.position;
		});
	renderer.run();*/

	constexpr size_t NUM_INPUTS = 2;
	constexpr int64_t NUM_LOOPS = 1;

	DataGenerator generator(pendulum, integrator);

	auto makeNetwork = [=](const nn::HyperParams& _params)
	{
		const size_t numInputsNet = _params.get<size_t>("num_inputs", NUM_INPUTS) * 2;

		nn::MultiLayerPerceptron net( numInputsNet, numInputsNet, numInputsNet-2, 
			_params.get<int>("depth", 32), 
			_params.get<bool>("bias", false));
		/*nn::AntiSymmetric net(numInputsNet,
			_params.get<int>("depth", 32),
			_params.get<double>("diffusion", 0.001),
			_params.get<double>("time", 10.0),
			_params.get<bool>("bias", false),
			_params.get<nn::ActivationFn>("activation", torch::tanh));*/
			//nn::AntiSymmetric net = getAntiSymmetric01();
		/*nn::HamiltonianAugmented net(nn::HamiltonianOptions(numInputsNet)
			.num_layers(_params.get<int>("depth", 32))
			.total_time(_params.get<double>("time", 4.0))
			.bias(_params.get<bool>("bias", false))
			.activation(torch::tanh)
			.augment_size(_params.get<int>("augment", 2)));*/
		/*nn::HamiltonianNet net(numInputsNet,
			_params.get<int>("depth", 32),
			_params.get<double>("time", 10.0),
			_params.get<bool>("bias", false),
			_params.get<nn::ActivationFn>("activation", torch::tanh));*/

		net.to(torch::kDouble);

		return net;
	};

	auto bestNet = makeNetwork(nn::HyperParams());

	auto trainNetwork = [=, &bestNet](const nn::HyperParams& _params)
	{
		const size_t numInputs = _params.get<size_t>("num_inputs", NUM_INPUTS);
		auto dataset = generator.generate({ state, state2, state3, state4 }, 128, 100, numInputs, false, NUM_LOOPS)
			.map(torch::data::transforms::Stack<>());
		auto validationSet = generator.generate({ validState1 }, 128, 100, numInputs, false, NUM_LOOPS)
			.map(torch::data::transforms::Stack<>());

		auto data_loader = torch::data::make_data_loader(
			dataset,
			torch::data::DataLoaderOptions().batch_size(64));
		auto validationLoader = torch::data::make_data_loader(
			validationSet,
			torch::data::DataLoaderOptions().batch_size(64));

		
		auto net = makeNetwork(_params);
		

		torch::optim::Adam optimizer(net.parameters(), 
			torch::optim::AdamOptions(_params.get<double>("lr", 1.e-4))
				.weight_decay(_params.get<double>("weight_decay", 1.e-6))
				.amsgrad(_params.get<bool>("amsgrad", false)));

		double bestLoss = std::numeric_limits<double>::max();

		//std::ofstream lossFile("loss.txt");

		for (int64_t epoch = 1; epoch <= 1024; ++epoch)
		{
			torch::Tensor totalLoss = torch::zeros({ 1 });
			for (torch::data::Example<>& batch : *data_loader)
			{
				torch::Tensor output = batch.data;
				for(int64_t i = 0; i < NUM_LOOPS; ++i)
					output = net.forward(output);
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
		{ {"depth", {42, 64}},
		  {"time", { 5.0, 7.0}},
	//	  {"bias", {false, true}},
	//	  {"diffusion", {0.0, 1.0e-5, 1.0e-3, 0.1}},
		  {"num_inputs", {4ull, 8ull, 16ull}}
		  //{"activation", {nn::ActivationFn(torch::tanh), nn::ActivationFn(torch::relu), nn::ActivationFn(torch::sigmoid)}}
		});

//	hyperOptimizer.run(8);
	
	nn::HyperParams params;
	params["lr"] = 1e-04;
	params["weight_decay"] = 1e-6;
	params["depth"] = 16;
	params["diffusion"] = 0.5;
	params["bias"] = false;
	params["time"] = 3.0;
	params["num_inputs"] = NUM_INPUTS;
	params["name"] = std::string("linear.pt");

	std::cout << trainNetwork(params) << "\n";

/*	eval::checkLayerStability(bestNet.inputLayer);
	for(auto& layer : bestNet.hiddenLayers)
		eval::checkLayerStability(layer);
	eval::checkLayerStability(bestNet.outputLayer);*/
//	eval::checkModuleStability(bestNet);

	
	torch::serialize::InputArchive archive;
	archive.load_from(*params.get<std::string>("name"));
	bestNet = makeNetwork(params);
//	nn::AntiSymmetric savedNet = getAntiSymmetric01();
//	bestNet.to(c10::kDouble);
	bestNet.load(archive);
	evaluate<NUM_INPUTS>(bestNet);
}
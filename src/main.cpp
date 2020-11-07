#include "systems/pendulum.hpp"
#include "discretization.hpp"
#include "nn/mlp.hpp"
#include "nn/dataset.hpp"
#include "nn/nnintegrator.hpp"
#include "visualisation/renderer.hpp"

#include <type_traits>
#include <torch/torch.h>
#include <chrono>
#include <iostream>

template<typename System, typename Integrator, typename State>
auto runSimulation(const System& _system, 
	const Integrator& _integrator, 
	const State& _initialState,
	int _steps,
	int _subSteps = 1)
{
	typename System::State state{ _initialState };
	std::vector<System::State> results;
	results.reserve(_steps);
	results.push_back(state);

	for (int i = 0; i < _steps; ++i)
	{
		state = _integrator(state);
		if(i % _subSteps == 0)
			results.push_back(state);
	}

	return results;
}

template<typename System, typename Integrator, typename States>
nn::Dataset generateDataset(const System& _system,
	const Integrator& _integrator,
	const States& _initStates)
{
	assert(_initStates.size() > 1);

	torch::Tensor inputs;
	torch::Tensor outputs;

	for (auto& state : _initStates)
	{
		auto results = runSimulation(_system, _integrator, state, 100000, 100);
		const int64_t size = static_cast<int64_t>(results.size()) - 1;
		torch::Tensor in = torch::from_blob(results.data(), { size, 2 }, c10::TensorOptions(c10::ScalarType::Double));
		torch::Tensor out = torch::from_blob(&results[1], { size, 2 }, c10::TensorOptions(c10::ScalarType::Double));

		if (!inputs.defined())
		{
			inputs = in.clone();
			outputs = out.clone();
		}
		else
		{
			inputs = torch::cat({ inputs, in });
			outputs = torch::cat({ outputs, out });
		}
	}

	return { inputs, outputs };
}

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

	visual::PendulumRenderer renderer(0.1);
	renderer.addIntegrator([&, state]() mutable
		{
			state = neuralNet(state);
			return state.position;
		});
	renderer.run();

	for (int i = 0; i < 64; ++i)
	{
		state1 = leapFrog(state1);
		state2 = forwardEuler(state2);
		state3 = neuralNet(state3);

	//	std::cout << state1.position << "; " << state1.velocity << std::endl;
	//	std::cout << state3.position << ", " << state3.velocity << std::endl;
	//	std::cout << pendulum.energy(state1) << ", " << pendulum.energy(state2) << ", " << pendulum.energy(state3) << "\n";
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

	State validState{ -1.3, 0.01 };

/*	visual::PendulumRenderer renderer(0.1);
	renderer.addIntegrator([&, state= state3]() mutable
		{
			state = integrator(state);
			return state.position;
		});
	renderer.run();*/

	auto dataset = generateDataset(pendulum, integrator, std::vector{ state, state2, state3 })
		.map(torch::data::transforms::Stack<>());
	auto data_loader = torch::data::make_data_loader(
		std::move(dataset),
		torch::data::DataLoaderOptions().batch_size(64));

	auto validationSet = generateDataset(pendulum, integrator, std::vector{ state, state2, state3 })
		.map(torch::data::transforms::Stack<>());
	auto validationLoader = torch::data::make_data_loader(
		std::move(validationSet),
		torch::data::DataLoaderOptions().batch_size(64));

	nn::MultiLayerPerceptron net(2, 2, 0, 0);
	nn::MultiLayerPerceptron bestNet;
	net.to(torch::kDouble);
	torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(0.0001));
	double bestLoss = std::numeric_limits<double>::max();

	for (int64_t epoch = 1; epoch <= 256; ++epoch)
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
		//	loss.backward();
			
			totalLoss += loss;

		//	optimizer.step();
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
		std::cout << "finished epoch with loss: " << totalLoss.item<double>() << "\n";
	}

	std::cout << "finished training!";
	evaluate(bestNet);

	/*
	for(int i = 0; i < 300 * 1000; ++i)
	{
		state = integrator(state);
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << state.position << ", " << state.velocity << std::endl;
	std::cout << std::chrono::duration<float>(end - start).count();*/
}
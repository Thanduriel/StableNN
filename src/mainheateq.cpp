#include "systems/heateq.hpp"
#include "systems/heateqsolver.hpp"
#include "nn/train.hpp"
#include "nn/mlp.hpp"
#include "nn/convolutional.hpp"
#include "nn/nnintegrator.hpp"
#include "evaluation/renderer.hpp"
#include "evaluation/evaluation.hpp"
#include <random>

constexpr size_t N = 64;
using System = systems::HeatEquation<double, N>;
using State = typename System::State;
using T = System::ValueT;

std::vector<State> generateStates(const System& _system, size_t _numStates, uint32_t _seed)
{
	std::vector<State> states;
	states.reserve(_numStates);

	std::default_random_engine rng(_seed);
	
	std::normal_distribution<T> energy(128.f, 64.f);

	for (size_t i = 0; i < _numStates; ++i)
	{
		const T e = energy(rng);
		State state;
		for (auto& v : state)
			v = std::max(static_cast<T>(0), energy(rng));
		states.push_back(state);
	}

	return states;
}

int main()
{
	System heatEq;

	auto trainingStates = generateStates(heatEq, 24, 0x612FF6AEu);
	auto validStates = generateStates(heatEq, 4, 0x195A4C);

	using Integrator = systems::discretization::AnalyticHeatEq<T, System::NumPoints>;
	using NetType = nn::Convolutional;
	nn::TrainNetwork<NetType, System, Integrator> trainNetwork(heatEq, trainingStates, validStates);

	nn::HyperParams params;
	params["hyper_sample_rate"] = 1;
	params["train_samples"] = 512;
	params["valid_samples"] = 1024;
	params["num_epochs"] = 1024;

	params["time_step"] = 0.0001;
	params["lr"] = 0.025;
	params["weight_decay"] = 1e-5; //4

	params["depth"] = 1;
	params["diffusion"] = 0.1;
	params["bias"] = false;
	params["time"] = 2.0;
	params["num_inputs"] = NUM_INPUTS;
	params["num_outputs"] = USE_SINGLE_OUTPUT ? 1 : NUM_INPUTS;
	params["augment"] = 2;
	params["hidden_size"] = static_cast<int>(N);
	params["filter_size"] = 33;
	params["train_in"] = false;
	params["train_out"] = false;
	params["activation"] = nn::ActivationFn(nn::identity);
	params["name"] = std::string("heateq64_conv")
		+ std::to_string(NUM_INPUTS) + "_"
		+ std::to_string(*params.get<int>("depth")) + "_"
		+ std::to_string(*params.get<int>("hidden_size"))
		+ ".pt";

	if constexpr (MODE == Mode::TRAIN_MULTI)
	{
		params["name"] = std::string("heateq64_conv");
		nn::GridSearchOptimizer hyperOptimizer(trainNetwork,
			{	{"filter_size", {3, 17, 33}},
				{"lr", {0.02, 0.025, 0.03}},
			//	{"activation", {nn::ActivationFn(torch::tanh), nn::ActivationFn(torch::relu), nn::ActivationFn(torch::sigmoid)}}
			}, params);

		hyperOptimizer.run(8);
	}

	if constexpr (MODE == Mode::TRAIN || MODE == Mode::TRAIN_EVALUATE)
		std::cout << trainNetwork(params) << "\n";
	if constexpr (MODE == Mode::EVALUATE)
	{
		const double timeStep = *params.get<double>("time_step");
		systems::discretization::AnalyticHeatEq analytic(heatEq, timeStep, validStates[0]);
		systems::discretization::FiniteDifferencesHeatEq finiteDiffs(heatEq, timeStep);

		auto net = nn::makeNetwork<NetType, USE_WRAPPER, 2>(params);
		torch::load(net, *params.get<std::string>("name"));
		nn::Integrator<System, decltype(net), NUM_INPUTS> nn(net);

		for (size_t i = 0; i < net->layers.size(); ++i)
			nn::exportTensor(net->layers[0]->weight, "heateq" + std::to_string(i) + ".txt");

		eval::HeatRenderer renderer(timeStep*100.f, [&, state = validStates[0]]() mutable
			{
				state = nn(state);
				std::vector<double> exState;
				for (double d : state)
					exState.push_back(d);
				return exState;
			});
		renderer.run();

		eval::EvalOptions options;
		eval::evaluate(heatEq, validStates[0], options, analytic, finiteDiffs, nn);
	}
}
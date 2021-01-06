#include "systems/heateq.hpp"
#include "systems/heateqsolver.hpp"
#include "nn/train.hpp"
#include "nn/mlp.hpp"
#include "nn/nnintegrator.hpp"
#include "evaluation/renderer.hpp"
#include "evaluation/evaluation.hpp"
#include <random>

using System = systems::HeatEquation<double, 64>;
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
//	systems::discretization::FiniteDifferencesHeatEq integ(heatEq, 0.0001);
/*	systems::discretization::AnalyticHeatEq integ(heatEq, 0.0001);
	System::State testState{};
	testState.fill(50.f);
	testState[32] = 272.0;
	testState[4] = 0.0;
	testState[5] = 0.0;
	testState[6] = 0.0;
	testState[42] = 78.0;

	int counter = 0;
	eval::HeatRenderer renderer(0.01, [&, state=testState]() mutable
		{
			state = integ(state);
			std::vector<double> exState;
			for (double d : state)
				exState.push_back(d);
			return exState;
		});
	renderer.run();*/

	auto trainingStates = generateStates(heatEq, 24, 0x612FF6AEu);
	auto validStates = generateStates(heatEq, 4, 0x195A4C);

	using Integrator = systems::discretization::AnalyticHeatEq<T, System::NumPoints>;
	using NetType = nn::MultiLayerPerceptron;
	nn::TrainNetwork<NetType, System, Integrator> trainNetwork(heatEq, trainingStates, validStates);

	nn::HyperParams params;
	params["hyper_sample_rate"] = 1;
	params["train_samples"] = 512;
	params["valid_samples"] = 1024;

	params["time_step"] = 0.0002;
	params["lr"] = 0.085;
	params["weight_decay"] = 1e-5; //4
	params["depth"] = 1;
	params["diffusion"] = 0.1;
	params["bias"] = false;
	params["time"] = 2.0;
	params["num_inputs"] = NUM_INPUTS;
	params["num_outputs"] = USE_SINGLE_OUTPUT ? 1 : NUM_INPUTS;
	params["augment"] = 2;
	params["hidden_size"] = 64;
	params["train_in"] = false;
	params["train_out"] = false;
	params["activation"] = nn::ActivationFn(nn::identity);
	params["name"] = std::string("heateq")
		+ std::to_string(NUM_INPUTS) + "_"
		+ std::to_string(*params.get<int>("depth")) + "_"
		+ std::to_string(*params.get<int>("hidden_size"))
		+ ".pt";

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
#include "systems/pendulum.hpp"
#include "systems/odesolver.hpp"
#include "systems/serialization.hpp"
#include "constants.hpp"
#include "defs.hpp"
#include "nn/train.hpp"
#include "nn/mlp.hpp"
#include "nn/dataset.hpp"
#include "nn/nnintegrator.hpp"
#include "nn/hyperparam.hpp"
#include "generator.hpp"
#include "nn/antisymmetric.hpp"
#include "nn/hamiltonian.hpp"
#include "nn/inoutwrapper.hpp"
#include "nn/utils.hpp"
#include "nn/nnmaker.hpp"
#include "evaluation/evaluation.hpp"
#include "evaluation/renderer.hpp"
#include "evaluation/stability.hpp"
#include "evaluation/lipschitz.hpp"
#include "evaluation/asymptotic.hpp"

#include <type_traits>
#include <torch/torch.h>
#include <chrono>
#include <iostream>
#include <cmath>
#include <random>
#include <filesystem>

using System = systems::Pendulum<double>;
using State = typename System::State;

template<size_t NumTimeSteps, typename... Networks>
void evaluate(
	const System& system,
	const State& _initialState, 
	double _timeStep, 
	eval::EvalOptions _options, 
	Networks&... _networks)
{
	namespace discret = systems::discretization;
	discret::LeapFrog<System> leapFrog(system, _timeStep);
	discret::ForwardEuler<System> forwardEuler(system, _timeStep);

	auto referenceIntegrate = [&](const State& _state)
	{
		discret::LeapFrog<System> forward(system, _timeStep / HYPER_SAMPLE_RATE);
		auto state = _state;
		for (int64_t i = 0; i < HYPER_SAMPLE_RATE; ++i)
			state = forward(state);
		return state;
	};

	// prepare initial time series
	const auto& [initialStates, initialState] = nn::computeTimeSeries<NumTimeSteps>(referenceIntegrate, _initialState);

	if constexpr (SHOW_VISUAL)
	{
		// copy integrator because nn::Integrator may have an internal state
		auto integrators = std::make_tuple(nn::Integrator<System, Networks, NumTimeSteps>(system, _networks, initialStates)...);
		auto visState = initialState;
		// warmup to see long term behavior
		for (int i = 0; i < 10000; ++i)
		{
			visState = std::get<0>(integrators)(visState);
		}

		eval::PendulumRenderer renderer(_timeStep);
		renderer.addIntegrator([drawIntegrator=std::get<0>(integrators), state=visState]() mutable
			{
				state = drawIntegrator(state);
				return state.position;
			});
		renderer.run();
	}

/*	auto cosRef = [&, t=0.0](const State& _state) mutable
	{
		t += _timeStep;
		return State{ std::cos(t / 2.30625 * 2.0 * PI) * _initialState.position, 0.0 };
	};*/
	
	eval::evaluate(system,
		initialState, 
		_options,
		referenceIntegrate, 
		leapFrog, 
		nn::Integrator<System, Networks, NumTimeSteps>(system, _networks, initialStates)...);
}

template<size_t NumTimeSteps, typename... Networks>
void evaluate(const System& _system, 
	const std::vector<State>& _initialStates, 
	double _timeStep,
	const eval::EvalOptions& _options,
	Networks&... _networks)
{
	for (const State& state : _initialStates)
		evaluate<NumTimeSteps>(_system, state, _timeStep, _options, _networks...);
}

template<size_t NumTimeSteps, typename... Networks>
void makeEnergyErrorData(const System& _system, double _timeStep, Networks&... _networks)
{
	eval::EvalOptions options;
	options.writeMSE = true;
	options.numLongTermSteps = 0;
	options.numShortTermSteps = 256;

	constexpr int numStates = 128;
	std::vector<State> states;
	states.reserve(numStates);
	for (int i = 0; i < numStates; ++i)
		states.push_back({ static_cast<double>(i) / numStates * PI, 0.0 });

	evaluate<NumTimeSteps>(_system, states, _timeStep, options, _networks...);
}

template<typename NetType>
void makeStableFrequencyData(const System& system, const nn::HyperParams& params)
{
	//{ 0.1, 0.05, 0.025, 0.01, 0.005 };
	std::vector<double> timeSteps = /*{ 0.1, 0.05, 0.025, 0.01, 0.005 };*/{ 0.05, 0.049, 0.048, 0.047, 0.046, 0.045 };
	std::vector<std::pair<std::string, double>> names;
	for (int j = 4; j < 6; ++j)
		for (int i = 0; i < 4; ++i)
		{
			const std::string name = std::to_string(i) + "_" + std::to_string(j) + "_closeFreq.pt";
			if (std::filesystem::exists(name))
				names.emplace_back(name, timeSteps[j]);
		}

	std::mutex outputMutex;
	auto computeFrequencies = [&](size_t begin, size_t end)
	{
		for (size_t i = begin; i < end; ++i)
		{
			const auto& [name, timeStep] = names[i];
			auto param = params;
			param["time_step"] = timeStep;
			auto othNet = nn::makeNetwork<NetType, USE_WRAPPER, 2>(param);
			torch::load(othNet, name);

			nn::Integrator<System, decltype(othNet), NUM_INPUTS> integrator(system, othNet);
			auto [attractors, repellers] = eval::findAttractors(system, integrator, false);
			std::vector<double> stablePeriods;
			for (double attractor : attractors)
			{
				if (attractor == 0.0 || attractor == eval::INF_ENERGY)
					continue;
				stablePeriods.push_back(
					eval::computePeriodLength(system.energyToState(0.0, attractor), integrator, 64, 6.0 / timeStep)
					* timeStep);
			}
			std::unique_lock lock(outputMutex);
			std::cout << name << "," << timeStep << ", ";
			for (double p : stablePeriods)
				std::cout << p << ",";
			std::cout << "\n";
		}
	};
	std::vector<std::thread> threads;
	const size_t numThreads = std::min(static_cast<size_t>(8), names.size());
	const size_t numTasks = names.size() / numThreads;
	for (size_t i = 0; i < numThreads - 1; ++i)
	{
		threads.emplace_back(computeFrequencies, i * numTasks, (i + 1) * numTasks);
	}
	computeFrequencies((numThreads - 1) * numTasks, names.size());
	for (auto& t : threads) t.join();
}

std::vector<State> generateStates(const System& _system, size_t _numStates, uint32_t _seed)
{
	std::vector<State> states;
	states.reserve(_numStates);

	std::default_random_engine rng(_seed);
	constexpr double MAX_POS = PI - 0.05;
	State maxState{ MAX_POS, 0.0 };
	const double maxEnergy = _system.energy(maxState);
	std::uniform_real_distribution<double> energy(0, maxEnergy);
	std::uniform_real_distribution<double> potEnergy(0.0, 1.0);
	std::bernoulli_distribution sign;

	for (size_t i = 0; i < _numStates; ++i)
	{
		const double e = energy(rng);
		const double potE = potEnergy(rng);
		State s = _system.energyToState((1.0 - potE) * e, potE * e);
		states.push_back({ sign(rng) ? s.position : -s.position, sign(rng) ? s.velocity : -s.velocity });
	}

	return states;
}

int main()
{
/*	auto nn_inp = torch::ones({3,2});
	auto state = torch::zeros({ 3,2,4 });
	state = torch::cat({ state.slice(2, 1), nn_inp.unsqueeze(2) }, 2);
	nn_inp *= 2.0;
	state = torch::cat({ state.slice(2, 1), nn_inp.unsqueeze(2) }, 2);
	std::cout << state;
	return 0;*/

	System system(0.1, 9.81, 0.5);

	auto trainingStates = generateStates(system, 180, 0x612FF6AEu);
	trainingStates.push_back({ 3.0,0 });
	trainingStates.push_back({ -2.9,0 });
	auto validStates = generateStates(system, 31, 0x195A4C);
	validStates.push_back({ 2.95,0 });
/*	for (auto& state : trainingStates)
	{
		Integrator integrator(system, TARGET_TIME_STEP);
		eval::PendulumRenderer renderer(TARGET_TIME_STEP);
		renderer.addIntegrator([&integrator, s=state] () mutable
			{
				s = integrator(s);
				return s.position;
			});
		renderer.run();
	}*/
	using Integrator = systems::discretization::LeapFrog<System>;
	using NetType = nn::HamiltonianInterleafed;
	nn::TrainNetwork<NetType, System, Integrator> trainNetwork(system, trainingStates, validStates);

	nn::HyperParams params;
	params["train_samples"] = 16;
	params["valid_samples"] = 16;

	params["time_step"] = 0.05;
	params["lr"] = 0.085;//4e-4;
	params["weight_decay"] = 1e-6; //4
	params["depth"] = 2;
	params["diffusion"] = 0.1;
	params["bias"] = false;
	params["time"] = 2.0;
	params["num_inputs"] = NUM_INPUTS;
	params["num_outputs"] = USE_SINGLE_OUTPUT ? 1 : NUM_INPUTS;
	params["augment"] = 2;
	params["hidden_size"] = HIDDEN_SIZE;
	params["train_in"] = true;
	params["train_out"] = false;
	params["in_out_bias"] = false;
	params["activation"] = nn::ActivationFn(torch::tanh);
	params["name"] = std::string("mlp_t0025_")
		+ std::to_string(NUM_INPUTS) + "_"
		+ std::to_string(*params.get<int>("depth")) + "_"
		+ std::to_string(*params.get<int>("hidden_size"))
		+ ".pt";

/*	auto testFn = [](const nn::HyperParams& params)
	{
		return *params.get<double>("1") + *params.get<double>("10") + *params.get<double>("100");
	};
	nn::GridSearchOptimizer hyperOptimizer(testFn, { {"1", {0.0, 1.0, 2.0}}, {"10", {0.0, 10.0}}, {"100", {0.0, 100.0}} });
	hyperOptimizer.run();
	return 0;*/

	if constexpr (MODE == Mode::TRAIN_MULTI)
	{
		params["name"] = std::string("hamiltonianInterleafed");
		nn::GridSearchOptimizer hyperOptimizer(trainNetwork,
			{	{"depth", {2, 4}},
			//  {"lr", {0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095}},
				{"lr", {0.08, 0.085}},
			//	{"weight_decay", {1e-6, 1e-5}},
				{"time", { 1.0, 2.0}},
			//	{"train_in", {false, true}},
			//	{"train_out", {false, true}},
			//  {"time_step", { 0.1, 0.05, 0.025, 0.01, 0.005 }},
			//	{"time_step", { 0.05, 0.049, 0.048, 0.047, 0.046, 0.045 }},
			//	{"hidden_size", {4, 8, 16}},
			//	{"bias", {false, true}},
			//	{"in_out_bias", {false,true}},
			//	{"diffusion", {0.08, 0.09, 0.1, 0.11, 0.12}},
			//	{"num_inputs", {4ull, 8ull, 16ull}}
				{"activation", {nn::ActivationFn(torch::tanh), nn::ActivationFn(nn::zerosigmoid), nn::ActivationFn(torch::sin)}}
			}, params);

		hyperOptimizer.run(8);
	}


	if constexpr (MODE == Mode::TRAIN || MODE == Mode::TRAIN_EVALUATE)
		std::cout << trainNetwork(params) << "\n";

	if constexpr (MODE == Mode::EVALUATE || MODE == Mode::TRAIN_EVALUATE)
	{
		eval::EvalOptions options;
	/*	nn::HyperParams hamiltonianParams(params);
		hamiltonianParams["train_in"] = true;
		hamiltonianParams["train_out"] = true;
		hamiltonianParams["time"] = 4.0;
		hamiltonianParams["name"] = std::string("hamiltonianIO.pt");
		auto hamiltonianIO = nn::makeNetwork<nn::HamiltonianInterleafed, true, 2>(hamiltonianParams);
		torch::load(hamiltonianIO, *hamiltonianParams.get<std::string>("name"));

		hamiltonianParams["train_in"] = false;
		//	hamiltonianParams["time"] = 2.0;
		hamiltonianParams["name"] = std::string("hamiltonianO.pt");
		auto hamiltonianO = nn::makeNetwork<nn::HamiltonianInterleafed, true, 2>(hamiltonianParams);
		torch::load(hamiltonianO, *hamiltonianParams.get<std::string>("name"));

		nn::HyperParams linearParams = params;
		linearParams["train_in"] = false;
		linearParams["train_out"] = false;
		linearParams["name"] = std::string("lin_1_4.pt");
		auto mlp = nn::makeNetwork<nn::MultiLayerPerceptron, true, 2>(linearParams);
		torch::load(mlp, *linearParams.get<std::string>("name"));

		nn::HyperParams antisymParams = params;
		antisymParams["train_in"] = false;
		antisymParams["train_out"] = true;
		antisymParams["name"] = std::string("antiSymO.pt");
		auto antiSym = nn::makeNetwork<nn::AntiSymmetric, true, 2>(antisymParams);
		torch::load(antiSym, *antisymParams.get<std::string>("name"));

		std::cout << eval::lipschitz(hamiltonianIO) << "\n";
		std::cout << eval::lipschitz(hamiltonianO) << "\n";
		std::cout << eval::lipschitz(mlp) << "\n";
		std::cout << eval::lipschitz(antiSym) << "\n";*/
		//	evaluate<NUM_INPUTS>(system, { { 0.5, 0.0 }, { 1.0, 0.0 }, { 1.5, 0.0 }, { 2.5, 0.0 }, { 3.0, 0.0 } }, 
		//		hamiltonianIO, hamiltonianO, mlp, antiSym);

		nn::HyperParams antisymParams = params;
		antisymParams["train_in"] = true;
		antisymParams["train_out"] = true;
		antisymParams["diffusion"] = 0.11;
		antisymParams["time"] = 3.0;
		antisymParams["name"] = std::string("1_3_antisym2.pt");
		auto antiSym = nn::makeNetwork<nn::AntiSymmetric, true, 2>(antisymParams);
		torch::load(antiSym, *antisymParams.get<std::string>("name"));

		nn::HyperParams resnetParams = params;
		resnetParams["train_in"] = true;
		resnetParams["name"] = std::string("resnet.pt");
		auto resNet = nn::makeNetwork<nn::MultiLayerPerceptron, USE_WRAPPER, 2>(params);

		auto othNet = nn::makeNetwork<NetType, USE_WRAPPER, 2>(params);
		torch::load(othNet, *params.get<std::string>("name"));
		nn::Integrator<System, decltype(othNet), NUM_INPUTS> integrator(system, othNet);
	//	auto [attractors, repellers] = eval::findAttractors(system, integrator, true);
		makeEnergyErrorData<NUM_INPUTS>(system, *params.get<double>("time_step"), othNet, antiSym);
	/*	evaluate<NUM_INPUTS>(system,
			{ { 0.5, 0.0 }, { 1.0, 0.0 }, { 1.5, 0.0 }, { 2.0, 0.0 }, { 2.5, 0.0 }, { 3.0, 0.0 } },
			* params.get<double>("time_step"),
			options,
			othNet);*/
		return 0;

		//	std::cout << eval::computeJacobian(othNet, torch::tensor({ 1.5, 0.0 }, c10::TensorOptions(c10::kDouble)));
		//	std::cout << eval::lipschitz(othNet) << "\n";
		//	std::cout << eval::lipschitzParseval(othNet->hiddenNet->hiddenLayers) << "\n";
		//	std::cout << eval::spectralComplexity(othNet->hiddenNet->hiddenLayers) << "\n";

	//	evaluate<NUM_INPUTS>(system, { {1.5, 1.0 }, { 0.9196, 0.0 }, { 0.920388, 0.0 }, { 2.2841, 0.0 }, { 2.28486, 0.0 } }, othNet);

	/*	for (size_t i = 0; i < othNet->hiddenNet->hiddenLayers.size(); ++i)
		{
			nn::exportTensor(othNet->hiddenNet->hiddenLayers[i]->weight, "layer" + std::to_string(i) + ".txt");
		}*/
	}
}
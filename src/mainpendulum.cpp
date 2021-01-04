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

using NetType = nn::MultiLayerPerceptron;

template<size_t NumTimeSteps, typename... Networks>
void evaluate(const System& system, const State& _initialState, double _timeStep, Networks&... _networks)
{
	State initialState{ _initialState };

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
	std::array<System::State, NumTimeSteps-1> initialStates;
	if (NumTimeSteps > 1)
	{
		initialStates[0] = initialState;
		for (size_t i = 1; i < initialStates.size(); ++i)
		{
			initialStates[i] = referenceIntegrate(initialStates[i - 1]);
		}
		initialState = referenceIntegrate(initialStates.back());
	}

	if constexpr (SHOW_VISUAL)
	{
		// copy integrator because nn::Integrator may have an internal state
		auto integrators = std::make_tuple(nn::Integrator<System, Networks, NumTimeSteps>(_networks, initialStates)...);
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
	
	eval::EvalOptions options;
	eval::evaluate(system,
		initialState, 
		options,
		referenceIntegrate, 
		leapFrog, 
		nn::Integrator<System, Networks, NumTimeSteps>(_networks, initialStates)...);
}

template<size_t NumTimeSteps, typename... Networks>
void evaluate(const System& _system, const std::vector<State>& _initialStates, double _timeStep, Networks&... _networks)
{
	for (const State& state : _initialStates)
		evaluate<NumTimeSteps>(_system, state, _timeStep, _networks...);
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
	nn::TrainNetwork<NetType, System, Integrator> trainNetwork(system, trainingStates, validStates);

	nn::HyperParams params;
	params["time_step"] = 0.05;
	params["lr"] = 0.085;//4e-4;
	params["weight_decay"] = 1e-6; //4
	params["depth"] = 4;
	params["diffusion"] = 0.1;
	params["bias"] = false;
	params["time"] = 2.0;
	params["num_inputs"] = NUM_INPUTS;
	params["num_outputs"] = USE_SINGLE_OUTPUT ? 1 : NUM_INPUTS;
	params["augment"] = 2;
	params["hidden_size"] = HIDDEN_SIZE;
	params["train_in"] = false;
	params["train_out"] = false;
	params["activation"] = nn::ActivationFn(torch::tanh);
	params["name"] = std::string("mlp_t0025_")
		+ std::to_string(NUM_INPUTS) + "_"
		+ std::to_string(*params.get<int>("depth")) + "_"
		+ std::to_string(*params.get<int>("hidden_size"))
		+ ".pt";

	if constexpr (MODE == Mode::TRAIN_MULTI)
	{
		params["name"] = std::string("closeFreq");
		nn::GridSearchOptimizer hyperOptimizer(trainNetwork,
			{ //{"depth", {4, 8}},
			//  {"lr", {0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095}},
				{"lr", {0.08, 0.081, 0.082, 0.083}},
			//  {"weight_decay", {1e-6}},
			//  {"time", { 2.0, 4.0, 3.0}},
			//  {"time_step", { 0.1, 0.05, 0.025, 0.01, 0.005 }},
				{"time_step", { 0.05, 0.049, 0.048, 0.047, 0.046, 0.045 }},
			  //	  {"hidden_size", {4, 8, 16}},
			  //	  {"bias", {false, true}},
			  //	  {"diffusion", {0.0, 0.1, 0.5}},
			  //	  {"num_inputs", {4ull, 8ull, 16ull}}
			  //	  {"activation", {nn::ActivationFn(torch::tanh), nn::ActivationFn(torch::relu), nn::ActivationFn(torch::sigmoid)}}
			}, params);

		hyperOptimizer.run(8);
	}


	if constexpr (MODE == Mode::TRAIN || MODE == Mode::TRAIN_EVALUATE)
		std::cout << trainNetwork(params) << "\n";

	if constexpr (MODE == Mode::EVALUATE || MODE == Mode::TRAIN_EVALUATE)
	{
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

		auto othNet = nn::makeNetwork<NetType, USE_WRAPPER, 2>(params);
		torch::load(othNet, *params.get<std::string>("name"));
		evaluate<NUM_INPUTS>(system,
			{ { 0.5, 0.0 }, { 1.0, 0.0 }, { 1.5, 0.0 }, { 2.0, 0.0 }, { 2.5, 0.0 }, { 3.0, 0.0 } },
			* params.get<double>("time_step"),
			othNet);
		return 0;

		//{ 0.1, 0.05, 0.025, 0.01, 0.005 };
		std::vector<double> timeSteps = /*{ 0.1, 0.05, 0.025, 0.01, 0.005 };*/{ 0.05, 0.049, 0.048, 0.047, 0.046, 0.045 };
		std::vector<std::pair<std::string, double>> names;
		for (int j = 4; j < 6; ++j)
			for(int i = 0; i < 4; ++i)
			{
				const std::string name = std::to_string(i) + "_" + std::to_string(j) + "_closeFreq.pt";
				if (std::filesystem::exists(name))
					names.emplace_back(name, timeSteps[j]);
			}
	//	names.erase(names.begin(), names.begin() + 35);
	//	names.erase(names.begin(), names.begin() + 6);
	//	names.erase(names.begin() + 36, names.end());
		// = { "0_.pt", "1_.pt", "2_.pt", "3_.pt", "4_.pt" };
		//{ 0.1, 0.05, 0.01, 0.005 };

	//	names.clear();
	//	names.emplace_back(*params.get<std::string>("name"), 0.05);
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
				//	evaluate<NUM_INPUTS>(system, { system.energyToState(0.0, 0.981318) }, timeStep, othNet);

				/*	evaluate<NUM_INPUTS>(system,
							{{ 0.5, 0.0 }, { 1.0, 0.0 }, { 1.5, 0.0 }, { 2.0, 0.0 }, { 2.5, 0.0 }, { 3.0, 0.0 } },
							*params.get<double>("time_step"),
							othNet);*/
				nn::Integrator<System, decltype(othNet), NUM_INPUTS> integrator(othNet);
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
		for (size_t i = 0; i < numThreads-1; ++i)
		{
			threads.emplace_back(computeFrequencies, i* numTasks, (i+1) * numTasks);
		}
		computeFrequencies((numThreads - 1) * numTasks, names.size());
		for (auto& t : threads) t.join();
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
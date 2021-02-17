#include "systems/heateq.hpp"
#include "systems/heateqsolver.hpp"
#include "nn/train.hpp"
#include "nn/mlp.hpp"
#include "nn/convolutional.hpp"
#include "nn/nnintegrator.hpp"
#include "evaluation/renderer.hpp"
#include "evaluation/evaluation.hpp"
#include "evaluation/stability.hpp"
#include <random>
#include <chrono>

constexpr bool USE_WRAPPER = false;
constexpr bool USE_LOCAL_DIFFUSIFITY = true;
constexpr size_t N = 32;
using System = systems::HeatEquation<double, N>;
using State = typename System::State;
using T = System::ValueT;

namespace disc = systems::discretization;

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

std::vector<System> generateSystems(size_t _numSystems, uint32_t _seed)
{
	std::vector<System> systems;
	systems.reserve(_numSystems);

	std::default_random_engine rng(_seed);
	constexpr T maxChange = 0.05;
	std::uniform_real_distribution<T> base(0.1, 2.0);
	for (size_t i = 0; i < _numSystems; ++i)
	{
		const T avg = base(rng);
		std::uniform_real_distribution<T> dist(-avg*0.9, +avg*0.9);

		std::array<T, N> coefs;
		for (size_t j = 0; j < N; ++j)
			coefs[j] = avg + dist(rng);

		// smoothing until neighboring points have a max distance of maxChange
		bool isSmooth = true;
		do {
			isSmooth = true;
			for (size_t j = 0; j < N; ++j)
			{
				const T next = coefs[(j+1) % N];
				const T da = coefs[j] - next;
				if (std::abs(da) > maxChange)
					isSmooth = false;
				coefs[j] = 0.75 * coefs[j] + 0.125 * next + 0.125 * coefs[(j - 1 + N) % N];
			}
		} while (!isSmooth);

		systems.emplace_back(coefs);
	}

	return systems;
}

int main()
{
	std::array<T, N> heatCoefs{};
	heatCoefs.fill(1.0);
	if (USE_LOCAL_DIFFUSIFITY)
	{
		for (size_t i = 0; i < N / 2; ++i)
		{
			heatCoefs[i] = 0.75 + static_cast<double>(i) / N;
			heatCoefs[N-i-1] = 0.75 + static_cast<double>(i) / N;
		}
	}
	System heatEq(heatCoefs, 1.0);

	using InputMaker = std::conditional_t<USE_LOCAL_DIFFUSIFITY,
		systems::discretization::MakeInputHeatEq,
		nn::StateToTensor>;

	using NetType = nn::Convolutional;

	nn::HyperParams params;
	params["time_step"] = 0.0001;
	params["hyper_sample_rate"] = USE_LOCAL_DIFFUSIFITY ? 64 : 1;
	params["train_samples"] = 256;
	params["valid_samples"] = 256;
	params["batch_size"] = 512;
	params["num_epochs"] = USE_LBFGS ? 256 : 2048;
	params["loss_p"] = 3;

	params["lr"] = USE_LBFGS ? 0.05 : 0.001;
	params["lr_decay"] = USE_LBFGS ? 1.0 : 1.0;
	params["weight_decay"] = 0.0;

	params["depth"] = 3;
	params["bias"] = true;
	params["num_inputs"] = NUM_INPUTS;
	params["num_outputs"] = USE_SINGLE_OUTPUT ? 1 : NUM_INPUTS;
	params["num_channels"] = USE_LOCAL_DIFFUSIFITY ? 2 : 1;
	params["augment"] = 2;
	params["hidden_size"] = N;
	params["hidden_channels"] = 4;
	params["kernel_size"] = 3;
	params["residual"] = true;
	params["train_in"] = false;
	params["train_out"] = false;
	params["activation"] = nn::ActivationFn(torch::tanh);
	params["name"] = std::string("heateq64_conv")
		+ std::to_string(NUM_INPUTS) + "_"
		+ std::to_string(*params.get<int>("depth")) + "_"
		+ std::to_string(*params.get<int>("hidden_size"));

	if constexpr (MODE != Mode::EVALUATE)
	{
		auto trainingStates = generateStates(heatEq, 64, 0x612FF6AEu);
		auto validStates = generateStates(heatEq, 8, 0x195A4Cu);
		auto warmupSteps = std::vector<size_t>{ 0, 64, 0, 256, 16, 4, 128, 2 };

		using Integrator = std::conditional_t<USE_LOCAL_DIFFUSIFITY,
			systems::discretization::SuperSampleIntegrator<T, N, N * 32>,
			systems::discretization::AnalyticHeatEq<T, N>>;

		auto systems = USE_LOCAL_DIFFUSIFITY ? generateSystems(trainingStates.size() + validStates.size(), 0x6341241)
			: std::vector<System>{ heatEq };
		
		nn::TrainNetwork<NetType, System, Integrator, InputMaker, USE_WRAPPER> trainNetwork(
			systems, trainingStates, validStates, warmupSteps);

		if constexpr (MODE == Mode::TRAIN_MULTI)
		{
			params["name"] = std::string("single_output_kernel2");
			nn::GridSearchOptimizer hyperOptimizer(trainNetwork,
				{//	{"kernel_size", {5}},
					{"hidden_channels", {4,8}},
				//	{"residual", {false, true}},
				//	{"bias", {false, true}},
					{"depth", {2,4}},
				//	{"lr", {0.02, 0.025, 0.03}},
				//	{"lr", {0.015, 0.01, 0.005}},
				//	{"lr_decay", {0.995, 0.994, 0.993}},
				//	{"amsgrad", {false, true}},
				//	{"num_epochs", {2048}},
				//	{ "momentum", {0.5, 0.6, 0.7} },
				//	{ "dampening", {0.5, 0.4, 0.3} },
				//	{"activation", {nn::ActivationFn(torch::tanh), nn::ActivationFn(nn::elu), nn::ActivationFn(torch::relu)}}
				}, params);

			hyperOptimizer.run(2);
		}
		if constexpr (MODE == Mode::TRAIN || MODE == Mode::TRAIN_EVALUATE)
			std::cout << trainNetwork(params) << "\n";
	}

	if constexpr (MODE == Mode::EVALUATE || MODE == Mode::TRAIN_EVALUATE)
	{
		auto validStates = generateStates(heatEq, 4, 0x195A4C);
		auto& state = validStates[0];
		auto trainingStates = generateStates(heatEq, 32, 0x612FF6AEu);

		auto system = USE_LOCAL_DIFFUSIFITY ? generateSystems(1, 0xE312A41)[0]
			: heatEq;

		const double timeStep = *params.get<double>("time_step");
		disc::AnalyticHeatEq analytic(system, timeStep, state);
		disc::FiniteDifferencesExplicit<T, N, 2> finiteDiffs(system, timeStep);
		disc::FiniteDifferencesImplicit<T, N, 2> finiteDiffsImpl(system, timeStep);
		disc::SuperSampleIntegrator<T, N, N * 32> superSampleFiniteDifs(system, timeStep, state, 64);

		auto net = nn::load<NetType, USE_WRAPPER>(params);
	//	torch::load(net, "0_1_0_1_diffusivity2.pt");
		nn::Integrator<System, decltype(net), NUM_INPUTS, InputMaker> nn(system, net);

	//	nn::exportTensor(analytic.getGreenFn(timeStep, 63), "green.txt");
	//	eval::checkEnergy(net->layers.front(), 64);
	//	for (size_t i = 0; i < net->layers.size(); ++i)
	//		nn::exportTensor(net->layers[i]->weight, "heateq_adam2" + std::to_string(i) + ".txt");

		if constexpr (SHOW_VISUAL)
		{
			auto systems = generateSystems(36, 0x6341241);
			for (int i = 0; i < 4; ++i) {
			//	disc::FiniteDifferencesExplicit finiteDiffs2(systems[i], timeStep);
				disc::SuperSampleIntegrator<T, N, N * 32> finiteDiffs2(systems[i], timeStep, trainingStates[i], 64);
				eval::HeatRenderer renderer(timeStep, N, systems[i].heatCoefficients().data(), [&, state = trainingStates[i]]() mutable
				{
					state = finiteDiffs2(state);
					return std::vector<double>(state.begin(), state.end());
				});
				renderer.run();
			}
		}

		eval::EvalOptions options;
		options.numShortTermSteps = 128;
	/*	for(auto& state : trainingStates)
		{
			disc::AnalyticHeatEq analytic(system, timeStep, state);
			disc::SuperSampleIntegrator<T, N, N * 8> superSampleFiniteDifs(system, timeStep, state, 4);
			disc::SuperSampleIntegrator<T, N, N * 16> superSampleFiniteDifs1(system, timeStep, state, 16);
			disc::SuperSampleIntegrator<T, N, N * 32> superSampleFiniteDifs2(system, timeStep, state, 64);

			disc::AnalyticHeatEq analyticLargeIntern(superSampleFiniteDifs2.internalSystem(), timeStep, superSampleFiniteDifs2.internalState());
			auto analytic2 = [&](const State&)
			{
				auto state = analyticLargeIntern({});
				return superSampleFiniteDifs2.downscaleState(state);
			};

			eval::evaluate(heatEq, state, options, analytic, finiteDiffs, superSampleFiniteDifs, superSampleFiniteDifs1, superSampleFiniteDifs2);
			break;
		}*/
		if constexpr (USE_LOCAL_DIFFUSIFITY)
			eval::evaluate(system, state, options, superSampleFiniteDifs, finiteDiffs, analytic, nn);
		else
			eval::evaluate(system, state, options, analytic, superSampleFiniteDifs, finiteDiffs, finiteDiffsImpl, nn);
	}
}
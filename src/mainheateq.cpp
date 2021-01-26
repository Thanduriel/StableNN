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

constexpr bool USE_LOCAL_DIFFUSIFITY = false;
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

std::vector<System> generateSystems(size_t _numSystems, uint32_t _seed)
{
	std::vector<System> systems;
	systems.reserve(_numSystems);

	std::default_random_engine rng(_seed);
	constexpr T maxChange = 0.05;
	std::uniform_real_distribution<T> base(0.5, 3.0);
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

	auto trainingStates = generateStates(heatEq, 24, 0x612FF6AEu);
	auto validStates = generateStates(heatEq, 3, 0x195A4C);

	using Integrator = std::conditional_t<USE_LOCAL_DIFFUSIFITY, 
		systems::discretization::SuperSampleIntegrator<T,N, N*4>,
		systems::discretization::AnalyticHeatEq<T, System::NumPoints>>;

	using InputMaker = std::conditional_t<USE_LOCAL_DIFFUSIFITY,
		systems::discretization::MakeInputHeatEq,
		nn::StateToTensor>;

	using NetType = nn::Convolutional;
	nn::TrainNetwork<NetType, System, Integrator, InputMaker> trainNetwork(heatEq, trainingStates, validStates);

	nn::HyperParams params;
	params["hyper_sample_rate"] = 1;
	params["train_samples"] = 512;
	params["valid_samples"] = 1024;
	params["batch_size"] = 1024;
	params["num_epochs"] = 1024;
	params["num_channels"] = USE_LOCAL_DIFFUSIFITY ? 2 : 1;

	params["time_step"] = 0.0001;
	params["lr"] = 0.002;
	params["weight_decay"] = 0.0;

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
	params["name"] = std::string("heateq64_conv_adam")
		+ std::to_string(NUM_INPUTS) + "_"
		+ std::to_string(*params.get<int>("depth")) + "_"
		+ std::to_string(*params.get<int>("hidden_size"))
		+ ".pt";

	if constexpr (MODE == Mode::TRAIN_MULTI)
	{
		params["name"] = std::string("temp");
		nn::GridSearchOptimizer hyperOptimizer(trainNetwork,
			{//	{"filter_size", {3, 17, 33}},
			//	{"lr", {0.02, 0.025, 0.03}},
				{"lr", {0.004, 0.003, 0.002}},
			//	{"amsgrad", {false, true}},
				{"num_epochs", {256,512, 1024}},
			//	{ "momentum", {0.5, 0.6, 0.7} },
			//	{ "dampening", {0.5, 0.4, 0.3} },
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
		systems::discretization::FiniteDifferencesExplicit finiteDiffs(heatEq, timeStep);
		systems::discretization::FiniteDifferencesImplicit finiteDiffsImpl(heatEq, timeStep);

		systems::discretization::SuperSampleIntegrator<T, N, N*2> superSampleFiniteDifs(heatEq, timeStep, 2, validStates[0]);

		auto net = nn::makeNetwork<NetType, USE_WRAPPER, 2>(params);
	//	torch::load(net, *params.get<std::string>("name"));
		torch::load(net, "2_0_heateq64_conv.pt");
		nn::Integrator<System, decltype(net), NUM_INPUTS, InputMaker> nn(heatEq, net);

		eval::checkEnergy(net->layers.front(), 64);
	//	for (size_t i = 0; i < net->layers.size(); ++i)
	//		nn::exportTensor(net->layers[i]->weight, "heateq" + std::to_string(i) + ".txt");

		auto systems = generateSystems(4, 0xF21B50C);
		systems::discretization::FiniteDifferencesExplicit finiteDiffs2(systems[2], timeStep);
		eval::HeatRenderer renderer(timeStep, N, systems[2].heatCoefficients().data(), [&, state = validStates[0]]() mutable
			{
				state = finiteDiffs2(state);
				std::vector<double> exState(state.begin(), state.end());
				return exState;
			});
		renderer.run();

		eval::EvalOptions options;
		eval::evaluate(heatEq, validStates[0], options, analytic, finiteDiffs, finiteDiffsImpl, superSampleFiniteDifs/*, nn*/);
	}
}
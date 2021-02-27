#include "systems/pendulum.hpp"
#include "systems/odesolver.hpp"
#include "systems/rungekutta.hpp"
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
#include "nn/tcn.hpp"
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
using NetType = nn::HamiltonianInterleafed;
constexpr bool USE_WRAPPER = !std::is_same_v<NetType, nn::TCN>;
// simulation related
constexpr int HYPER_SAMPLE_RATE = 128;
constexpr bool USE_SIMPLE_SYSTEM = true;

namespace discret = systems::discretization;
using LeapFrog = discret::ODEIntegrator<System, discret::LeapFrog>;

template<size_t NumTimeSteps, typename... Networks>
void evaluate(
	const System& system,
	const State& _initialState, 
	double _timeStep, 
	eval::EvalOptions _options, 
	Networks&... _networks)
{
	LeapFrog leapFrog(system, _timeStep);
	discret::ODEIntegrator<System, discret::ForwardEuler> forwardEuler(system, _timeStep);
	discret::ODEIntegrator<System, discret::RungeKutta<discret::RK2_midpoint>> rk4(system, _timeStep);

	auto referenceIntegrate = [&](const State& _state)
	{
		LeapFrog forward(system, _timeStep / HYPER_SAMPLE_RATE);
		auto state = _state;
		for (int i = 0; i < HYPER_SAMPLE_RATE; ++i)
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
void makeEnergyErrorData(const System& _system, double _timeStep, int _numSteps, Networks&... _networks)
{
	eval::EvalOptions options;
	options.writeMSE = true;
	options.numLongTermSteps = 0;
	options.numShortTermSteps = _numSteps;

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
			auto othNet = nn::makeNetwork<NetType, USE_WRAPPER>(param);
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

template<typename Network>
void makeJacobianData(
	Network& _network, 
	const State& _min,
	const State& _max, 
	const systems::Vec<double, 2>& _steps)
{
	using namespace torch;
	std::ofstream file1("spectral_radius.txt");
	std::ofstream file2("determinant.txt");
	for (double v = _min.velocity; v <= _max.velocity; v += _steps[1])
	{
		for (double p = _min.position; p <= _max.position; p += _steps[0])
		{
			const Tensor z = tensor({ p, v }, c10::TensorOptions(c10::kDouble));
			const Tensor J = eval::computeJacobian(_network, z);
			const auto eigenvalues = eval::computeEigs(J);
			double max = 0.0;
			std::complex<double> det = 1.0;
			for (auto eig : eigenvalues)
			{
				const double abs = std::abs(eig);
				if (abs > max) max = abs;
				det *= eig;
			}
			file1 << p << " " << v << " " << max << "\n";
			file2 << p << " " << v << " " << det.real() << "\n";
		}
		file1 << "\n";
		file2 << "\n";
	}
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
	System system = USE_SIMPLE_SYSTEM ? System(1.0, 1.0, 1.0) : System(0.1, 9.81, 0.5);

	auto trainingStates = generateStates(system, 180, 0x612FF6AEu);
	trainingStates.push_back({ 3.0,0 });
	trainingStates.push_back({ -2.9,0 });
	auto validStates = generateStates(system, 63, 0x195A4C);
	validStates.push_back({ 2.95,0 });
/*	for (auto& state : trainingStates)
	{
		LeapFrog integrator(system, USE_SIMPLE_SYSTEM ? 0.25 : 0.05);
		eval::PendulumRenderer renderer(0.5);
		renderer.addIntegrator([&integrator, s=state] () mutable
			{
				s = integrator(s);
				return s.position;
			});
		renderer.run();
	}*/
	using Integrator = LeapFrog;
	using TrainNetwork = nn::TrainNetwork<NetType, System, Integrator, nn::MakeTensor_t<NetType>, nn::StateToTensor, USE_WRAPPER>;
	TrainNetwork trainNetwork(system, trainingStates, validStates);

	nn::HyperParams params;
	params["train_samples"] = 16;
	params["valid_samples"] = 16;
	params["hyper_sample_rate"] = HYPER_SAMPLE_RATE;

	params["time_step"] = USE_SIMPLE_SYSTEM ? 0.25 : 0.05;
	params["lr"] = USE_LBFGS ? 0.1 : 4e-4;//0.001;//4e-4;
	params["weight_decay"] = 0.0; //4
	params["loss_p"] = 3;
	params["lr_decay"] = USE_LBFGS ? 1.0 : 0.2; // 0.998 : 0.999
	params["lr_epoch_update"] = 800;
	params["batch_size"] = 64;
	params["num_epochs"] = USE_LBFGS ? 256 : 1024;
	params["momentum"] = 0.9;
	params["dampening"] = 0.0;

	params["seed"] = 9378341130ull;
	params["depth"] = 2;
	params["diffusion"] = 0.05;
	params["bias"] = false;
	params["time"] = 0.25;
	params["num_inputs"] = NUM_INPUTS;
	params["num_outputs"] = USE_SINGLE_OUTPUT ? 1 : NUM_INPUTS;
	params["state_size"] = systems::sizeOfState<System>();
	params["hidden_size"] = 8;
	params["train_in"] = false;
	params["train_out"] = false;
	params["in_out_bias"] = false;
	params["activation"] = nn::ActivationFn(torch::tanh);

	params["augment"] = 4;
	params["kernel_size"] = 3;
	params["residual_blocks"] = 2;
	params["block_size"] = 2;
	params["average"] = true;
	params["num_channels"] = systems::sizeOfState<System>();

	params["name"] = std::string("antisym_2_8");
	params["load_net"] = false;

/*	std::uniform_int_distribution<uint64_t> dist(std::numeric_limits<uint64_t>::min(), std::numeric_limits<uint64_t>::max());
	std::default_random_engine rng;
	for (int i = 0; i < 8; ++i)
		std::cout << dist(rng) << "ull,";*/

	if constexpr (MODE == Mode::TRAIN_MULTI)
	{
		params["name"] = std::string("HamiltonianAugmented_2_4l");
		nn::GridSearchOptimizer hyperOptimizer(trainNetwork,
			{//	{"depth", {2, 4}},
			//  {"lr", {0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095}},
			//	{"lr", {0.005, 0.001, 0.0005}},
			//	{"lr_decay", {1.0, 0.9995, 0.999}},
			//	{"batch_size", {64, 256}},
			//	{"num_epochs", {4096, 8000}},
			//	{"weight_decay", {1e-6, 1e-5}},
				{"time", { 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0 }},
			//	{"train_in", {false, true}},
			//	{"train_out", {false, true}},
			//  {"time_step", { 0.1, 0.05, 0.025, 0.01, 0.005 }},
			//	{"time_step", { 0.05, 0.049, 0.048, 0.047, 0.046, 0.045 }},
			//	{"bias", {false, true}},
			//	{"in_out_bias", {false,true}},
			//	{"diffusion", {0.0, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0}},
			//	{"hidden_size", {4, 8}},
			//	{"num_inputs", {8}},
			//	{"kernel_size", {3, 5}},
			//	{"residual_blocks", {1,2,3}},
			//	{"residual", {false, true}},
			//	{"average", {false, true}},
			//	{"block_size", {1,2,3}},
			//	{"activation", {nn::ActivationFn(torch::tanh), nn::ActivationFn(nn::zerosigmoid), nn::ActivationFn(nn::elu)}},
			//	{"seed", {9378341130ul, 16708911996216745849ull, 2342493223442167775ull, 16848810653347327969ull, 11664969248402573611ull, 1799302827895858725ull, 5137385360522333466ull, 10088183424363624464ull}}
			}, params);

		hyperOptimizer.run(8);
	}

	if constexpr (MODE == Mode::TRAIN || MODE == Mode::TRAIN_EVALUATE)
	{
		trainNetwork(params);
	}

	if constexpr (MODE == Mode::EVALUATE || MODE == Mode::TRAIN_EVALUATE)
	{
	/*	eval::EvalOptions options;

		auto othNet = nn::load<NetType, USE_WRAPPER>(params);
	//	nn::exportTensor(othNet->layers->at<torch::nn::LinearImpl>(othNet->layers->size()-1).weight, "outlinear.txt");
		nn::Integrator<System, decltype(othNet), NUM_INPUTS> integrator(system, othNet);
	//	auto [attractors, repellers] = eval::findAttractors(system, integrator, true);
	//	makeEnergyErrorData<NUM_INPUTS>(system, *params.get<double>("time_step"), othNet);
		evaluate<NUM_INPUTS>(system,
			{ { 0.5, 0.0 }, { 1.0, 0.0 }, { 1.5, 0.0 }, { 2.0, 0.0 }, { 2.5, 0.0 }, { 3.0, 0.0 } },
			*params.get<double>("time_step"),
			options,
			othNet);
		return 0;*/

	//	Integrator integrator(system, *params.get<double>("time_step"));
	//	std::cout << eval::computePeriodLength(State{ 0.01 * PI, 0.0 }, integrator, 16);

		const double energy = system.energy({ 3.0,0 });
		const auto state = system.energyToState(energy * 0.75, energy * 0.25);
		std::cout << state;
		auto mlp = nn::load<nn::MultiLayerPerceptron, USE_WRAPPER>(params, "resnet_2_4l");
		makeJacobianData(mlp, { -2.0, -1.0 }, { 2.0, 1.01 }, { 0.05, 0.05 });

	/*	auto mlpIO = nn::load<nn::MultiLayerPerceptron, USE_WRAPPER>(params, "mlpIO");
		auto antisym = nn::load<nn::AntiSymmetric, USE_WRAPPER>(params, "antisym");
		auto hamiltonian = nn::load<nn::HamiltonianInterleafed, USE_WRAPPER>(params, "hamiltonian");*/
	/*	auto mlpIO = nn::load<nn::MultiLayerPerceptron, USE_WRAPPER>(params, "resnet_2_4l");
		auto antisym = nn::load<nn::AntiSymmetric, USE_WRAPPER>(params, "antisym_2_4l");
		auto antisym2 = nn::load<nn::AntiSymmetric, USE_WRAPPER>(params, "antisym_2_4");
		auto hamiltonian = nn::load<nn::HamiltonianInterleafed, USE_WRAPPER>(params, "HamiltonianInterleafed_2_4l");
		makeEnergyErrorData<NUM_INPUTS>(system, *params.get<double>("time_step"), 8, mlpIO, antisym, hamiltonian);*/

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
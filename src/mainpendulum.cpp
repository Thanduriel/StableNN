#include "nn/train.hpp"
#include "nn/mlp.hpp"
#include "nn/dataset.hpp"
#include "generator.hpp"
#include "nn/antisymmetric.hpp"
#include "nn/hamiltonian.hpp"
#include "nn/tcn.hpp"
#include "nn/inoutwrapper.hpp"
#include "nn/utils.hpp"
#include "nn/nnmaker.hpp"
#include "pendulumeval.hpp"

#include <torch/torch.h>
#include <chrono>
#include <cmath>
#include <random>

using NetType = nn::MultiLayerPerceptron;
constexpr bool USE_WRAPPER = !std::is_same_v<NetType, nn::TCN>;

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
	auto validStates = generateStates(system, 63, 0x195A4Cu);
	validStates.push_back({ 2.95,0 });

	using Integrator = LeapFrog;
	using TrainNetwork = nn::TrainNetwork<NetType, System, Integrator, nn::MakeTensor_t<NetType>, nn::StateToTensor, USE_WRAPPER>;
	TrainNetwork trainNetwork(system, trainingStates, validStates);

	nn::HyperParams params;
	params["name"] = std::string("pendulum_net");
	params["load_net"] = false;
	params["net_type"] = std::string(typeid(NetType).name());

	// data
	params["train_samples"] = 16;
	params["valid_samples"] = 16;
	params["hyper_sample_rate"] = HYPER_SAMPLE_RATE;

	// training
	params["time_step"] = USE_SIMPLE_SYSTEM ? 0.25 : 0.05;
	params["lr"] = USE_LBFGS ? 0.1 : 4e-4;//0.001;//4e-4;
	params["weight_decay"] = 0.0; //4
	params["loss_p"] = 3;
	params["loss_factor"] = 100.0;
	params["lr_decay"] = USE_LBFGS ? 1.0 : 0.1; // 0.998 : 0.999
	params["lr_epoch_update"] = 2048;
	params["batch_size"] = 64;
	params["num_epochs"] = USE_LBFGS ? 512 : 2048;
	params["momentum"] = 0.9;
	params["dampening"] = 0.0;
	params["seed"] = static_cast<uint64_t>(16708911996216745849ull);

	// network
	params["depth"] = 2;
	params["bias"] = false;
	params["time"] = 2.0;
	params["num_inputs"] = NUM_INPUTS;
	params["num_outputs"] = USE_SINGLE_OUTPUT ? 1 : NUM_INPUTS;
	params["state_size"] = systems::sizeOfState<System>();
	params["hidden_size"] = 4;
	params["train_in"] = true;
	params["train_out"] = true;
	params["in_out_bias"] = false;
	params["activation"] = nn::ActivationFn(torch::tanh);

	// Antisym
	params["diffusion"] = 0.025;

	// Hamiltonian
	params["augment"] = 4;
	params["symmetric"] = false;
	
	// TCN
	params["residual_blocks"] = 2;
	params["block_size"] = 2;
	params["average"] = true;
	params["kernel_size"] = 3;
	params["num_channels"] = systems::sizeOfState<System>();

	if constexpr (MODE == Mode::TRAIN_MULTI)
	{
		constexpr int numSeeds = 8;
		std::mt19937_64 rng;
		std::vector<nn::ExtAny> seeds(numSeeds);
		std::generate(seeds.begin(), seeds.end(), rng);

		nn::GridSearchOptimizer hyperOptimizer(trainNetwork,
			{	{"depth", {2, 4}},
			//  {"lr", {0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095}},
			//	{"lr", {0.005, 0.001, 0.0005}},
			//	{"lr_decay", {1.0, 0.9995, 0.999}},
			//	{"batch_size", {64, 256}},
			//	{"num_epochs", {4096, 8000}},
			//	{"weight_decay", {1e-6, 1e-5}},
			//	{"time", { 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0 }},
			//	{"time", { 1.0, 2.0 }},
			//	{"train_in", {false, true}},
			//	{"train_out", {false, true}},
			//  {"time_step", { 0.1, 0.05, 0.025, 0.01, 0.005 }},
			//	{"time_step", { 0.05, 0.049, 0.048, 0.047, 0.046, 0.045 }},
			//	{"bias", {false, true}},
			//	{"in_out_bias", {false,true}},
			//	{"diffusion", {0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0}},
			//	{"hidden_size", {4, 8}},
			//	{"num_inputs", {8}},
			//	{"kernel_size", {3, 5}},
			//	{"residual_blocks", {1,2,3}},
			//	{"residual", {false, true}},
			//	{"average", {false, true}},
			//	{"block_size", {1,2,3}},
			//	{"activation", {nn::ActivationFn(torch::tanh), nn::ActivationFn(nn::zerosigmoid), nn::ActivationFn(nn::elu)}},
				{"seed", seeds}
			}, params);

		const unsigned numThreads = std::thread::hardware_concurrency();
		hyperOptimizer.run(std::max(1u, numThreads / 2));
	}

	if constexpr (MODE == Mode::TRAIN || MODE == Mode::TRAIN_EVALUATE)
	{
		trainNetwork(params);
	}

	if constexpr (MODE == Mode::EVALUATE || MODE == Mode::TRAIN_EVALUATE)
	{
		const double timeStep = *params.get<double>("time_step");
		eval::EvalOptions options;

		auto net = nn::load<NetType, USE_WRAPPER>(params);
		evaluate<NUM_INPUTS>(system,
			{ { 0.5, 0.0 }, { 1.0, 0.0 }, { 1.5, 0.0 }, { 2.0, 0.0 }, { 2.5, 0.0 }, { 3.0, 0.0 } },
			timeStep,
			options,
			net);
	}
}
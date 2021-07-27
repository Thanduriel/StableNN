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
/*	for (auto& state : trainingStates)
	{
		LeapFrog integrator(system, USE_SIMPLE_SYSTEM ? 0.25 : 0.05);
		eval::PendulumRenderer renderer(0.5, [&integrator, s=state] () mutable
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
	// data
	params["train_samples"] = 16;
	params["valid_samples"] = 16;
	params["hyper_sample_rate"] = HYPER_SAMPLE_RATE;
	// training
	params["time_step"] = USE_SIMPLE_SYSTEM ? 0.25 : 0.05;
	params["lr"] = USE_LBFGS ? 0.1 : 4e-4;//0.001;//4e-4;
	params["weight_decay"] = 0.0; //4
	params["loss_p"] = 3;
	params["lr_decay"] = USE_LBFGS ? 1.0 : 0.1; // 0.998 : 0.999
	params["lr_epoch_update"] = 2048;
	params["batch_size"] = 64;
	params["num_epochs"] = USE_LBFGS ? 512 : 2048;
	params["momentum"] = 0.9;
	params["dampening"] = 0.0;
	params["seed"] = 16708911996216745849ull;//9378341130ull;

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

	// Antisym and Hamiltonian
	params["diffusion"] = 0.025;
	params["augment"] = 4;
	params["symmetric"] = false;
	params["kernel_size"] = 3;
	// TCN
	params["residual_blocks"] = 2;
	params["block_size"] = 2;
	params["average"] = true;
	params["num_channels"] = systems::sizeOfState<System>();

	params["name"] = std::string("antisym_4_4");
	params["load_net"] = false;
	params["net_type"] = nn::sanitizeString(std::string(typeid(NetType).name()));

/*	std::uniform_int_distribution<uint64_t> dist(std::numeric_limits<uint64_t>::min(), std::numeric_limits<uint64_t>::max());
	std::default_random_engine rng;
	for (int i = 0; i < 8; ++i)
		std::cout << dist(rng) << "ull,";*/

//	makeLipschitzData<NetType>(params, trainNetwork);
//	return 0;

	if constexpr (MODE == Mode::TRAIN_MULTI)
	{
		params["name"] = std::string("hamiltonianSym_2_4l");
		nn::GridSearchOptimizer hyperOptimizer(trainNetwork,
			{//	{"depth", {2, 4}},
			//  {"lr", {0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095}},
			//	{"lr", {0.005, 0.001, 0.0005}},
			//	{"lr_decay", {1.0, 0.9995, 0.999}},
			//	{"batch_size", {64, 256}},
			//	{"num_epochs", {4096, 8000}},
			//	{"weight_decay", {1e-6, 1e-5}},
			//	{"time", { 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0 }},
				{"time", { 1.0, 2.0 }},
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
				{"seed", {9378341130ul, 16708911996216745849ull, 2342493223442167775ull, 16848810653347327969ull, 11664969248402573611ull, 1799302827895858725ull, 5137385360522333466ull, 10088183424363624464ull}}
			}, params);

		hyperOptimizer.run(2);
	}

	if constexpr (MODE == Mode::TRAIN || MODE == Mode::TRAIN_EVALUATE)
	{
		trainNetwork(params);
	}

	if constexpr (MODE == Mode::EVALUATE || MODE == Mode::TRAIN_EVALUATE)
	{
		const double timeStep = *params.get<double>("time_step");
		eval::EvalOptions options;
	//	options.numLongTermRuns = 8;

	/*	auto othNet = nn::load<NetType, USE_WRAPPER>(params);
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

		auto resNet24l = nn::load<nn::MultiLayerPerceptron, USE_WRAPPER>(params, "resnet_4_2l");
		auto resNet44l = nn::load<nn::MultiLayerPerceptron, USE_WRAPPER>(params, "resnet_4_4l");
		auto resNet26l = nn::load<nn::MultiLayerPerceptron, USE_WRAPPER>(params, "resnet_2_6l");

		makeMultiSimAvgErrorData<NUM_INPUTS>(system, timeStep, 2048, 4, 2.8, resNet24l, resNet26l, resNet44l);
		return 0;
	//	Integrator integrator(system, *params.get<double>("time_step"));
	//	std::cout << eval::computePeriodLength(State{ 0.0001, 0.0 }, integrator, 16);

	/*	auto mlpIORef = nn::load<nn::MultiLayerPerceptron, USE_WRAPPER>(params, "resnet_2_4l");
		auto mlp = nn::load<nn::MultiLayerPerceptron, USE_WRAPPER>(params, "mlp_4_4l");
		auto mlpIO = nn::load<nn::MultiLayerPerceptron, USE_WRAPPER>(params, "mlp_4_4l_t2");
		auto antisym = nn::load<nn::AntiSymmetric, USE_WRAPPER>(params, "antisym_4_4l");
		auto hamiltonian = nn::load<nn::HamiltonianInterleaved, USE_WRAPPER>(params, "hamiltonian_4_4l");*/
	/*	auto mlp = nn::load<nn::MultiLayerPerceptron, USE_WRAPPER>(params, "mlp");
		auto mlpIO = nn::load<nn::MultiLayerPerceptron, USE_WRAPPER>(params, "mlpIO");
		auto antisym = nn::load<nn::AntiSymmetric, USE_WRAPPER>(params, "antisym");
		auto hamiltonian = nn::load<nn::HamiltonianInterleaved, USE_WRAPPER>(params, "hamiltonian");*/
		auto mlpIO = nn::load<nn::MultiLayerPerceptron, USE_WRAPPER>(params, "resnet_4_4l");
	/*	auto antisym = nn::load<nn::AntiSymmetric, USE_WRAPPER>(params, "antisym_2_4l");
		auto antisym2 = nn::load<nn::AntiSymmetric, USE_WRAPPER>(params, "antisym_2_4");
		auto antisym3 = nn::load<nn::AntiSymmetric, USE_WRAPPER>(params, "antisym_8_2");
		auto hamiltonian = nn::load<nn::HamiltonianInterleaved, USE_WRAPPER>(params, "HamiltonianInterleaved_2_4l");*/

	/*	auto antisym0d = nn::load<nn::AntiSymmetric, USE_WRAPPER>(params, "antisym_4_4_0d");
		auto antisym01d = nn::load<nn::AntiSymmetric, USE_WRAPPER>(params, "antisym_4_4_01d");
		auto antisym001d = nn::load<nn::AntiSymmetric, USE_WRAPPER>(params, "antisym_4_4");
		auto hamiltonianSym = nn::load<nn::HamiltonianInterleaved, USE_WRAPPER>(params, "hamiltonianSym_4_4");*/

		nn::Integrator<System, decltype(mlpIO), NUM_INPUTS> integrator(system, mlpIO);
		discret::ODEIntegrator<System, StaticResNet> resNetBench(system, timeStep);

	//	evaluate<NUM_INPUTS>(system, system.energyToState(1.02763, 0.0), timeStep, options, mlpIO/*, antisym, hamiltonian*/);
		while (true)
		{
			int numSteps;
			std::cin >> numSteps;

			makeRuntimeData(system, timeStep, numSteps, resNetBench);
		}
		return 0;
	/*	auto state = system.energyToState(1.02765, 0.0);
		double e = 0.0;
		int n = 100000;
		for (int i = 0; i < n; ++i)
		{
			state = integrator(state);
			e += system.energy(state);
		}
		std::cout << e / n;*/

		std::string netName = "resnet_4_4l";
		auto net = nn::load<nn::MultiLayerPerceptron, USE_WRAPPER>(params, netName);
		nn::exportTensor(net->inputLayer->weight, "input_" + netName + ".txt");
		nn::exportTensor(net->outputLayer->weight, "output_" + netName + ".txt");
		nn::exportTensor(net->hiddenNet->layers[0]->weight, "layer0_" + netName + ".txt", false);
		nn::exportTensor(net->hiddenNet->layers[1]->weight, "layer1_" + netName + ".txt", false);
		nn::exportTensor(net->hiddenNet->layers[2]->weight, "layer2_" + netName + ".txt", false);
		nn::exportTensor(net->hiddenNet->layers[3]->weight, "layer3_" + netName + ".txt", false);
		return 0;

		discret::ODEIntegrator<System, discret::RungeKutta<discret::RK3_ssp>> rk3(system, timeStep);
		discret::ODEIntegrator<System, discret::RungeKutta<discret::RK4>> rk4(system, timeStep);
		LeapFrog leapFrog(system, timeStep);
		nn::VerletPendulum verletNN(timeStep, HYPER_SAMPLE_RATE);
	//	auto [attractors, repellers] = eval::findAttractors(system, leapFrog, true);

		makeSinglePhasePeriodData<NUM_INPUTS>(system, State{ 1.1309, 0.0 }, timeStep, mlpIO);
		return 0;

	//	eval::EvalOptions options;
	//	options.numLongTermRuns = 128;
	//	options.writeGlobalError = true;
		options.numShortTermSteps = 128;
		options.mseAvgWindow = 64;
		options.downSampleRate = 32;// 1.02765 
		return 0;
	//	evaluate<NUM_INPUTS>(system, /*State{ -0.324639, 1.39669 }*/system.energyToState(1.02763, 0.0), timeStep, options, mlpIO/*, antisym, hamiltonian*/);
	//	makeGlobalErrorData(system, system.energyToState(1.02763, 0.0), timeStep, mlpIO, antisym, hamiltonian);
	//	makeEnergyErrorData<NUM_INPUTS>(system, timeStep, 1024, 16, antisym0d, antisym01d, antisym001d);
	//	makeVerletPeriodErrorData(system, timeStep);
	//	makeAsymptoticEnergyData<NUM_INPUTS>(system, integrator);
	//	const double maxEnergy = system.energy(State{ 3.1, 0.0 });
	//	auto restrictDomain = [&](const State& _state) { return system.energy(_state) > maxEnergy; };
		auto restrictNone = [&](const State& _state) { return false; };
		makeJacobianData(verletNN, { -PI, -2.0 }, { PI, 2.0 }, { 64, 64 }, restrictNone);
		return 0;

	/*	std::cout << eval::lipschitz(mlp) << ", "
			<< eval::lipschitz(mlpIO) << ", " 
			<< eval::lipschitz(antisym) << ", " 
			<< eval::lipschitz(hamiltonian) << "\n";*/

		//	std::cout << eval::computeJacobian(othNet, torch::tensor({ 1.5, 0.0 }, c10::TensorOptions(c10::kDouble)));
		//	std::cout << eval::lipschitz(othNet) << "\n";
		//	std::cout << eval::lipschitzParseval(othNet->hiddenNet->hiddenLayers) << "\n";
		//	std::cout << eval::spectralComplexity(othNet->hiddenNet->hiddenLayers) << "\n";

	//	evaluate<NUM_INPUTS>(system, { {1.5, 1.0 }, { 0.9196, 0.0 }, { 0.920388, 0.0 }, { 2.2841, 0.0 }, { 2.28486, 0.0 } }, othNet);

	/*	auto& net = antisym;
		std::cout << eval::details::norm(net->inputLayer->weight, 2) << ", "
			<< eval::details::norm(net->outputLayer->weight, 2) << "\n";
		for (size_t i = 0; i < net->hiddenNet->layers.size(); ++i)
		{
			std::cout << eval::details::norm(net->hiddenNet->layers[i]->systemMatrix(), 2);
			auto eigs = eval::computeEigs(antisym3->hiddenNet->layers[i]->systemMatrix());
			for (auto eig : eigs)
				std::cout << eig << "\n";
			std::cout << "\n";
		//	nn::exportTensor(othNet->hiddenNet->hiddenLayers[i]->weight, "layer" + std::to_string(i) + ".txt");
		}*/
	}
}
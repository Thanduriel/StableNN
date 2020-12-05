#include "systems/pendulum.hpp"
#include "discretization.hpp"
#include "nn/mlp.hpp"
#include "nn/dataset.hpp"
#include "nn/nnintegrator.hpp"
#include "evaluation/renderer.hpp"
#include "evaluation/evaluation.hpp"
#include "nn/hyperparam.hpp"
#include "generator.hpp"
#include "nn/antisymmetric.hpp"
#include "nn/hamiltonian.hpp"
#include "nn/inoutwrapper.hpp"
#include "nn/utils.hpp"
#include "nn/nnmaker.hpp"
#include "evaluation/stability.hpp"
#include "evaluation/lipschitz.hpp"

#include <type_traits>
#include <torch/torch.h>
#include <chrono>
#include <iostream>
#include <cmath>
#include <random>

// simulation related
constexpr double TARGET_TIME_STEP = 0.05;
constexpr int64_t HYPER_SAMPLE_RATE = 100;

// network parameters
constexpr size_t NUM_INPUTS = 1;
constexpr bool USE_SINGLE_OUTPUT = true;
constexpr int HIDDEN_SIZE = 2 * 2;
constexpr bool USE_WRAPPER = USE_SINGLE_OUTPUT && (NUM_INPUTS > 1 || HIDDEN_SIZE > 2);

// training
constexpr bool TRAIN_NET = false;
constexpr int64_t NUM_FORWARDS = 1;
constexpr bool SAVE_NET = true;
constexpr bool USE_HYPER_OPTIMIZER = false;

// evaluation
constexpr bool SHOW_VISUAL = false;

using System = systems::Pendulum<double>;
using State = System::State;

template<size_t NumTimeSteps, typename... Networks>
void evaluate(const State& _initialState, Networks&... _networks)
{
	System system(0.1, 9.81, 0.5);
	State initialState{ _initialState };

	discretization::LeapFrog<System> leapFrog(system, TARGET_TIME_STEP);
	discretization::ForwardEuler<System> forwardEuler(system, TARGET_TIME_STEP);

	auto referenceIntegrate = [&](const System::State _state)
	{
		discretization::LeapFrog<System> forward(system, TARGET_TIME_STEP / HYPER_SAMPLE_RATE);
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
		eval::PendulumRenderer renderer(TARGET_TIME_STEP);

		// use extra integrator because nn::Integrator may have an internal state
		auto integrators = std::make_tuple(nn::Integrator<System, Networks, NumTimeSteps>(_networks, initialStates)...);
		for (int i = 0; i < 10000; ++i)
		{
			initialState = std::get<0>(integrators)(initialState);
		}
		renderer.addIntegrator([drawIntegrator=std::get<0>(integrators), state=initialState]() mutable
			{
				state = drawIntegrator(state);
				return state.position;
			});
		renderer.run();
	}

	eval::evaluate(system, initialState, referenceIntegrate, leapFrog, 
		nn::Integrator<System, Networks, NumTimeSteps>(_networks, initialStates)...);
}

template<size_t NumTimeSteps, typename... Networks>
void evaluate(const std::vector<State>& _initialStates, Networks&... _networks)
{
	for (const State& state : _initialStates)
		evaluate<NumTimeSteps>(state, _networks...);
}

std::vector<State> generateStates(const System& _system, size_t _numStates, uint32_t _seed)
{
	std::vector<State> states;
	states.reserve(_numStates);

	std::default_random_engine rng(_seed);
	constexpr double MAX_POS = 3.14159 - 0.14;
	State maxState{ MAX_POS, 0.0 };
	const double maxEnergy = _system.energy(maxState);
	std::uniform_real_distribution<double> energy(0, maxEnergy);
	std::uniform_real_distribution<double> potEnergy(0.0, 1.0);
	std::bernoulli_distribution sign;

	for (size_t i = 0; i < _numStates; ++i)
	{
		const double e = energy(rng);
		const double potE = potEnergy(rng);
		const double v = std::sqrt(2.0 * (1.0 - potE) * e / (_system.mass() * _system.length() * _system.length()));
		const double p = std::acos(1.0 - potE * e / (_system.mass() * _system.gravity() * _system.length()));
		states.push_back({ sign(rng) ? p : -p, sign(rng) ? v : -v });
	}

	return states;
}

template<typename Net>
void findAttractors(Net& _net)
{
	System system(0.1, 9.81, 0.5);
	nn::Integrator<System, Net, NUM_INPUTS> integrator(_net);

	std::vector<double> attractors; // energy
	std::vector<std::pair<double,double>> repellers; // position [lower, upper]

	auto integrate = [&](double x)
	{
		State state{ x, 0.0 };
		for (int i = 0; i < 20000; ++i)
			state = integrator(state);

		return state;
	};

	const double stepSize = 3.14159 / 16.0;
	constexpr double THRESHOLD = 0.05;
	for (double x = 0.0; x < 3.14159; x += stepSize)
	{
		State state = integrate(x);
		const double e = std::min(2.0, system.energy(state));

		if (attractors.empty() || std::abs(attractors.back() - e) > THRESHOLD)
		{
			std::cout << "new attractor " << e << ", " << x << std::endl;
			attractors.push_back(e);
			repellers.push_back({ 0.0, x });
		}
	/*	double e0 = 0.0;
		double e1 = system.energy(state);
		do {
			e1 = e0;
			state = integrator(state);
			double e0 = system.energy(state);
		} while (e0 - e1 > 0.1);*/
	}
	
	// refine repellers
	for (size_t j = 1; j < repellers.size(); ++j)
	{
		double x = repellers[j].second;
		double step = stepSize * 0.5;
		double sign = -1.0;
		for (int i = 0; i < 8; ++i)
		{
			x += step * sign;
			State state = integrate(x);
			step *= 0.5;
			const double e = std::min(2.0, system.energy(state));
			if (std::abs(e - attractors[j]) > THRESHOLD)
			{
				sign = 1.0;
				repellers[j].first = x;
			}
			else
			{
				sign = -1.0;
				repellers[j].second = x;
			}
		}
		std::cout << "\n====\n";
	}
	std::cout << "repellers: \n";
	for (auto& [min,max] : repellers)
	{
		std::cout << "[" << min << ", " << max << "]" << "\n";
	}
}

int main()
{
	System system(0.1, 9.81, 0.5);
	using Integrator = discretization::LeapFrog<systems::Pendulum<double>>;
	Integrator integrator(system, TARGET_TIME_STEP / HYPER_SAMPLE_RATE);

	auto trainingStates = generateStates(system, 256, 0x612FF6AEu);
	auto validStates = generateStates(system, 32, 0x195A4C);
	/*	for (auto& state : trainingStates)
		{
			eval::PendulumRenderer renderer(TARGET_TIME_STEP);
			renderer.addIntegrator([&, state] () mutable
				{
					state = integrator(state);
					return state.position;
				});
			renderer.run();
		}*/

	DataGenerator generator(system, integrator);
	using NetType = nn::MultiLayerPerceptron;

	auto trainNetwork = [=](const nn::HyperParams& _params)
	{
		const size_t numInputs = _params.get<size_t>("num_inputs", NUM_INPUTS);
		auto dataset = generator.generate(trainingStates, 16, HYPER_SAMPLE_RATE, numInputs, USE_SINGLE_OUTPUT, NUM_FORWARDS)
			.map(torch::data::transforms::Stack<>());
		auto validationSet = generator.generate(validStates, 16, HYPER_SAMPLE_RATE, numInputs, USE_SINGLE_OUTPUT, NUM_FORWARDS)
			.map(torch::data::transforms::Stack<>());

		auto data_loader = torch::data::make_data_loader(
			dataset,
			torch::data::DataLoaderOptions().batch_size(64));
		auto validationLoader = torch::data::make_data_loader(
			validationSet,
			torch::data::DataLoaderOptions().batch_size(64));


		auto net = nn::makeNetwork<NetType, USE_WRAPPER, 2>(_params);
		auto bestNet = nn::makeNetwork<NetType, USE_WRAPPER, 2>(_params);
		//	auto lossFn = [](const torch::Tensor& self, const torch::Tensor& target) { return torch::mse_loss(self, target); };
		auto lossFn = [](const torch::Tensor& self, const torch::Tensor& target) { return nn::lp_loss(self, target, 3); };

		auto nextInput = [](const torch::Tensor& input, const torch::Tensor& output)
		{
			return (USE_SINGLE_OUTPUT && NUM_FORWARDS > 1) ? nn::shiftTimeSeries(input, output, 2) : output;
		};

		torch::optim::Adam optimizer(net->parameters(),
			torch::optim::AdamOptions(_params.get<double>("lr", 3.e-4))
			.weight_decay(_params.get<double>("weight_decay", 1.e-6))
			.amsgrad(_params.get<bool>("amsgrad", false)));

		double bestValidLoss = std::numeric_limits<double>::max();

		//std::ofstream lossFile("loss.txt");

		for (int64_t epoch = 1; epoch <= 2048; ++epoch)
		{
			// train
			net->train();
			torch::Tensor totalLoss = torch::zeros({ 1 });
			for (torch::data::Example<>& batch : *data_loader)
			{
				net->zero_grad();
				torch::Tensor output;
				torch::Tensor input = batch.data;
				for (int64_t i = 0; i < NUM_FORWARDS; ++i)
				{
					output = net->forward(input);
					input = nextInput(input, output);
				}
				torch::Tensor loss = lossFn(output, batch.target);
				totalLoss += loss;

				loss.backward();
				optimizer.step();
			}

			// validation
			net->eval();
			torch::Tensor validLoss = torch::zeros({ 1 });
			for (torch::data::Example<>& batch : *validationLoader)
			{
				torch::Tensor output;
				torch::Tensor input = batch.data;
				for (int64_t i = 0; i < NUM_FORWARDS; ++i)
				{
					output = net->forward(input);
					input = nextInput(input, output);
				}
				torch::Tensor loss = lossFn(output, batch.target);
				validLoss += loss;
			}


			const double totalValidLossD = validLoss.item<double>();
			if (totalValidLossD < bestValidLoss)
			{
				if constexpr (!USE_HYPER_OPTIMIZER)
					std::cout << validLoss.item<double>() << "\n";
				bestNet = nn::clone(net);
				bestValidLoss = totalValidLossD;
			}

			//	lossFile << totalLoss.item<double>() << ", " << totalLossD << "\n";
			//	std::cout << "finished epoch with loss: " << totalLoss.item<double>() << "\n";
		}
		if (SAVE_NET)
		{
			torch::save(bestNet, _params.get<std::string>("name", "net.pt"));
		}

		return bestValidLoss;
	};

	nn::HyperParams params;
	params["lr"] = 2e-4;
	params["weight_decay"] = 1e-6; //4
	params["depth"] = 4;
	params["diffusion"] = 0.1;
	params["bias"] = false;
	params["time"] = 1.0;
	params["num_inputs"] = NUM_INPUTS;
	params["num_outputs"] = USE_SINGLE_OUTPUT ? 1 : NUM_INPUTS;
	params["augment"] = 4;
	params["hidden_size"] = HIDDEN_SIZE;
	params["activation"] = nn::ActivationFn(torch::tanh);
	params["name"] = std::string("lin_")
		+ std::to_string(NUM_INPUTS) + "_"
		+ std::to_string(*params.get<int>("depth")) + "_"
		+ std::to_string(*params.get<int>("hidden_size"))
		+ ".pt";

	nn::GridSearchOptimizer hyperOptimizer(trainNetwork,
		{ {"depth", {4, 8}},
		  {"lr", {2e-4, 4e-4}},
		//	  {"time", { 1.0, 2.0}},
		//	  {"hidden_size", {4, 8, 16}},
		//	  {"bias", {false, true}},
		//	  {"diffusion", {0.0, 1.0e-2, 0.1, 0.5}},
		//	  {"num_inputs", {4ull, 8ull, 16ull}}
			  //{"activation", {nn::ActivationFn(torch::tanh), nn::ActivationFn(torch::relu), nn::ActivationFn(torch::sigmoid)}}
		}, params);

	if constexpr (USE_HYPER_OPTIMIZER)
	{
		hyperOptimizer.run(8);
		return 0;
	}


	if (TRAIN_NET)
		std::cout << trainNetwork(params) << "\n";

	/*	eval::checkLayerStability(bestnet->inputLayer);
			eval::checkLayerStability(layer);
		eval::checkLayerStability(bestnet->outputLayer);*/
		//eval::checkModuleStability(bestNet);
	auto othNet = nn::makeNetwork<NetType, USE_WRAPPER, 2>(params);
	torch::load(othNet, "lin_1_4.pt");
	//torch::load(othNet, *params.get<std::string>("name"));

	std::cout << eval::lipschitzParseval(othNet->hiddenNet->hiddenLayers) << "\n";
	std::cout << eval::spectralComplexity(othNet->hiddenNet->hiddenLayers) << "\n";
//	findAttractors(othNet);
//	evaluate<NUM_INPUTS>({ {1.5, 1.0 }, { 0.9196, 0.0 }, { 0.920388, 0.0 }, { 2.2841, 0.0 }, { 2.28486, 0.0 } }, othNet);

	/*	for (size_t i = 0; i < othNet->hiddenNet->hiddenLayers.size(); ++i)
		{
			nn::exportTensor(othNet->hiddenNet->hiddenLayers[i]->weight, "layer" + std::to_string(i) + ".txt");
		}*/
		//nn::exportTensor(othNet->outputLayer->weight, "multiStep3.txt");

	/*	auto netL2 = makeNetwork<nn::MultiLayerPerceptronExt>(params);
		torch::load(netL2, "0_0_linext.pt");
		auto netL4 = makeNetwork<NetType>(params);
		torch::load(netL4, "0_0_.pt");
		params["depth"] = 16;
		auto netL3 = makeNetwork<NetType>(params);
		torch::load(netL3, "1_1_.pt");

		for (size_t i = 0; i < netL3->hiddenLayers.size(); ++i)
		{
			nn::exportTensor(netL3->hiddenLayers[i]->weight, "min" + std::to_string(i) + ".txt");
		}*/

		//	evaluate<NUM_INPUTS>({ { 0.5, 0.0 }, { 1.0, 0.0 }, { 1.5, 0.0 }, { 2.5, 0.0 }, { 3.0, 0.0 } }, netL2, netL3, netL4);
}
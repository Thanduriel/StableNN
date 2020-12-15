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
#include "evaluation/asymptotic.hpp"

#include <type_traits>
#include <torch/torch.h>
#include <chrono>
#include <iostream>
#include <cmath>
#include <random>

// simulation related
constexpr int64_t HYPER_SAMPLE_RATE = 100;

// network parameters
constexpr size_t NUM_INPUTS = 1;
constexpr bool USE_SINGLE_OUTPUT = true;
constexpr int HIDDEN_SIZE = 2 * 2;
constexpr bool USE_WRAPPER = USE_SINGLE_OUTPUT && (NUM_INPUTS > 1 || HIDDEN_SIZE > 2);

// training
enum struct Mode {
	TRAIN,
	EVALUATE,
	TRAIN_MULTI,
	TRAIN_EVALUATE
};
constexpr Mode MODE = Mode::EVALUATE;
constexpr int64_t NUM_FORWARDS = 1;
constexpr bool SAVE_NET = true;
using NetType = nn::MultiLayerPerceptron;

// evaluation
constexpr bool SHOW_VISUAL = false;

using System = systems::Pendulum<double>;
using State = System::State;

template<size_t NumTimeSteps, typename... Networks>
void evaluate(const System& system, const State& _initialState, double _timeStep, Networks&... _networks)
{
	State initialState{ _initialState };

	discretization::LeapFrog<System> leapFrog(system, _timeStep);
	discretization::ForwardEuler<System> forwardEuler(system, _timeStep);

	auto referenceIntegrate = [&](const State& _state)
	{
		discretization::LeapFrog<System> forward(system, _timeStep / HYPER_SAMPLE_RATE);
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
		eval::PendulumRenderer renderer(_timeStep);

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

/*	auto cosRef = [&, t=0.0](const State& _state) mutable
	{
		t += _timeStep;
		return State{ std::cos(t / 2.30625 * 2.0 * 3.14159) * _initialState.position, 0.0 };
	};*/

	eval::evaluate(system, initialState, referenceIntegrate, leapFrog, 
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
	constexpr double MAX_POS = 3.14159 - 0.05;
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
	//System system(1.0, 1.0, 1.0);

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

	auto trainNetwork = [&system, &trainingStates, &validStates](const nn::HyperParams& _params)
	{
		using Integrator = discretization::LeapFrog<systems::Pendulum<double>>;
		Integrator referenceIntegrator(system, *_params.get<double>("time_step") / HYPER_SAMPLE_RATE);
		DataGenerator generator(system, referenceIntegrator);

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
				if constexpr (MODE != Mode::TRAIN_MULTI)
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
	params["time_step"] = 0.05;
	params["lr"] = 4e-4;
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
	params["train_out"] = true;
	params["activation"] = nn::ActivationFn(torch::tanh);
	params["name"] = std::string("mlp_t0025_")
		+ std::to_string(NUM_INPUTS) + "_"
		+ std::to_string(*params.get<int>("depth")) + "_"
		+ std::to_string(*params.get<int>("hidden_size"))
		+ ".pt";

	if constexpr (MODE == Mode::TRAIN_MULTI)
	{
		nn::GridSearchOptimizer hyperOptimizer(trainNetwork,
			{ //{"depth", {4, 8}},
			  {"lr", {1e-4, 2e-4, 4e-4, 8e-4}},
			  {"weight_decay", {1e-6}},
			  //	  {"time", { 2.0, 4.0}},
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
		torch::load(othNet, "3_0_.pt");
	//	evaluate<NUM_INPUTS>(system, { system.energyToState(0.87829, 0.0) }, othNet);
	//	torch::load(othNet,*params.get<std::string>("name"));

		//std::cout << eval::computeJacobian(othNet, torch::tensor({ 1.5, 0.0 }, c10::TensorOptions(c10::kDouble)));
		//	std::cout << eval::lipschitz(othNet) << "\n";
		//	std::cout << eval::lipschitzParseval(othNet->hiddenNet->hiddenLayers) << "\n";
		//	std::cout << eval::spectralComplexity(othNet->hiddenNet->hiddenLayers) << "\n";
		nn::Integrator<System, decltype(othNet), NUM_INPUTS> integrator(othNet);
	//	std::cout << eval::computePeriodLength(system.energyToState(0.87829, 0.0), integrator, 64) * *params.get<double>("time_step") << "\n";
		evaluate<NUM_INPUTS>(system, 
			{ { 0.5, 0.0 }, { 1.0, 0.0 }, { 1.5, 0.0 }, { 2.0, 0.0 }, { 2.5, 0.0 }, { 3.0, 0.0 } }, 
			*params.get<double>("time_step"),
			othNet);

		auto [attractors, repellers] = eval::findAttractors(system, integrator);
		for (double attractor : attractors)
		{
			if (attractor == 0.0 || attractor == eval::INF_ENERGY)
				continue;
			std::cout << eval::computePeriodLength(system.energyToState(attractor, 0.0), integrator, 64) 
				* *params.get<double>("time_step") << "\n";
		}
	//	evaluate<NUM_INPUTS>(system, { {1.5, 1.0 }, { 0.9196, 0.0 }, { 0.920388, 0.0 }, { 2.2841, 0.0 }, { 2.28486, 0.0 } }, othNet);

	/*	for (size_t i = 0; i < othNet->hiddenNet->hiddenLayers.size(); ++i)
		{
			nn::exportTensor(othNet->hiddenNet->hiddenLayers[i]->weight, "layer" + std::to_string(i) + ".txt");
		}*/
	//nn::exportTensor(othNet->outputLayer->weight, "multiStep3.txt");


	}
}
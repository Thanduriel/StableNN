#include "systems/pendulum.hpp"
#include "systems/heateq.hpp"
#include "systems/odesolver.hpp"
#include "systems/heateqsolver.hpp"
#include "systems/serialization.hpp"
#include "constants.hpp"
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
constexpr Mode MODE = Mode::TRAIN_EVALUATE;
constexpr int64_t NUM_FORWARDS = 1;
constexpr bool SAVE_NET = true;
constexpr bool LOG_LOSS = true;
constexpr bool USE_LBFGS = true;
// only relevant in TRAIN_MULTI to enforce same initial rng state for all networks
constexpr bool THREAD_FIXED_SEED = true;
constexpr bool USE_SEQ_SAMPLER = true;
using NetType = nn::MultiLayerPerceptron;

// evaluation
constexpr bool SHOW_VISUAL = false;

using System = systems::Pendulum<double>;
using State = typename System::State;

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
		eval::PendulumRenderer renderer(_timeStep);

		// use extra integrator because nn::Integrator may have an internal state
		auto integrators = std::make_tuple(nn::Integrator<System, Networks, NumTimeSteps>(_networks, initialStates)...);
		// warmup to see long term behavior
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
/*	systems::HeatEquation<double, 64> heatEq;
//	systems::discretization::FiniteDifferencesHeatEq integ(heatEq, 0.0001);
	systems::discretization::AnalyticHeatEq integ(heatEq, 0.0001);
	systems::HeatEquation<double, 64>::State testState{};
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
	renderer.run();
	return 0;*/

	System system(0.1, 9.81, 0.5);
	//System system(1.0, 1.0, 1.0);

	constexpr uint64_t torchSeed = 9378341130ul;//9378341134ul;
	torch::manual_seed(torchSeed);

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

	std::mutex initMutex;
	std::mutex loggingMutex;
	auto trainNetwork = [&, torchSeed](const nn::HyperParams& _params)
	{
		using Integrator = systems::discretization::LeapFrog<System>;
		Integrator referenceIntegrator(system, *_params.get<double>("time_step") / HYPER_SAMPLE_RATE);
		DataGenerator generator(system, referenceIntegrator);

		namespace dat = torch::data;
		const size_t numInputs = _params.get<size_t>("num_inputs", NUM_INPUTS);
		auto dataset = generator.generate(trainingStates, 16, HYPER_SAMPLE_RATE, numInputs, USE_SINGLE_OUTPUT, NUM_FORWARDS)
			.map(dat::transforms::Stack<>());
		auto validationSet = generator.generate(validStates, 16, HYPER_SAMPLE_RATE, numInputs, USE_SINGLE_OUTPUT, NUM_FORWARDS)
			.map(dat::transforms::Stack<>());

		// LBFGS does not work with mini batches
		using Sampler = std::conditional_t<USE_SEQ_SAMPLER, 
			dat::samplers::SequentialSampler, 
			dat::samplers::RandomSampler>;
		auto data_loader = dat::make_data_loader<Sampler>(
			dataset,
			dat::DataLoaderOptions().batch_size(USE_LBFGS ? std::numeric_limits< size_t>::max() : 64));
		auto validationLoader = dat::make_data_loader(
			validationSet,
			dat::DataLoaderOptions().batch_size(64));

		if constexpr (THREAD_FIXED_SEED)
		{
			initMutex.lock();
			torch::manual_seed(torchSeed);
		}
		auto net = nn::makeNetwork<NetType, USE_WRAPPER, 2>(_params);
		auto bestNet = nn::makeNetwork<NetType, USE_WRAPPER, 2>(_params);
		if constexpr (THREAD_FIXED_SEED)
		{
			initMutex.unlock();
		}

	/*	if constexpr (USE_LBFGS)
		{
			for (auto& layer : net->hiddenNet->hiddenLayers)
			{
				torch::NoGradGuard guard;
			//	layer->weight.fill_(1.0);
			}
		}*/

		//	auto lossFn = [](const torch::Tensor& self, const torch::Tensor& target) { return torch::mse_loss(self, target); };
		auto lossFn = [](const torch::Tensor& self, const torch::Tensor& target) { return nn::lp_loss(self, target, 3); };

		auto nextInput = [](const torch::Tensor& input, const torch::Tensor& output)
		{
			if constexpr (USE_SINGLE_OUTPUT && NUM_FORWARDS > 1)
				return nn::shiftTimeSeries(input, output, 2);
			else
				return output;
		};

		
		auto makeOptimizer = [&_params, &net]()
		{
			if constexpr (USE_LBFGS)
				return torch::optim::LBFGS(net->parameters(),
					torch::optim::LBFGSOptions(*_params.get<double>("lr")));
			else
				return torch::optim::Adam(net->parameters(),
					torch::optim::AdamOptions(_params.get<double>("lr", 3.e-4))
					.weight_decay(_params.get<double>("weight_decay", 1.e-6))
					.amsgrad(_params.get<bool>("amsgrad", false)));
		};
		auto optimizer = makeOptimizer();

		double bestValidLoss = std::numeric_limits<double>::max();

		//std::ofstream lossFile("loss.txt");

		for (int64_t epoch = 1; epoch <= 2048; ++epoch)
		{
			// train
			net->train();
			
			torch::Tensor totalLoss = torch::zeros({ 1 });

			for (torch::data::Example<>& batch : *data_loader)
			{
				auto closure = [&]()
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
					return loss;
				};

				optimizer.step(closure);
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
		if(LOG_LOSS)
		{
			std::unique_lock<std::mutex> lock(loggingMutex);
			std::ofstream lossLog("losses.txt", std::ios::app);
			lossLog << bestValidLoss << std::endl;
		}
		if (SAVE_NET)
		{
			torch::save(bestNet, _params.get<std::string>("name", "net.pt"));
		}

		return bestValidLoss;
	};

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
		for (int i = 0; i < numThreads-1; ++i)
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
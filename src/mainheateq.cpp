#include "nn/train.hpp"
#include "nn/mlp.hpp"
#include "nn/nnintegrator.hpp"
#include "evaluation/renderer.hpp"
#include "evaluation/evaluation.hpp"
#include "evaluation/stability.hpp"
#include "heateqeval.hpp"
#include <random>
#include <chrono>

using NetType = nn::Convolutional;
static_assert((!std::is_same_v<NetType, nn::TCN2d> && !std::is_same_v<NetType, nn::ExtTCN>) || USE_LOCAL_DIFFUSIFITY, 
	"input tensor not implemented for this config");

// @param _randMeanDev add random offset to the given mean
// @param _normalize shift _state so that the mean is guaranteed to be _mean.
// @param _stdDev standard deviation for the normal distribution
std::vector<State> generateStates(const System& _system, 
	size_t _numStates, 
	uint32_t _seed, 
	T _randMeanDev = 0.0,
	T _mean = MEAN,
	T _stdDev = STD_DEV,
	bool _normalize = false)
{
	std::vector<State> states;
	states.reserve(_numStates);

	std::default_random_engine rng(_seed);
	T base = _mean;
	std::uniform_real_distribution<T> meanDev(-_randMeanDev, _randMeanDev);

	for (size_t i = 0; i < _numStates; ++i)
	{
		if (_randMeanDev != 0.0)
			base = _mean + meanDev(rng);
		std::normal_distribution<T> energy(base, _stdDev);
		auto genEnergy = [&]() 
		{
			if constexpr (ABSOLUTE_ZERO)
				return std::max(static_cast<T>(0), energy(rng));
			else
				return energy(rng);
		};

		State state;
		std::generate(state.begin(), state.end(), genEnergy);
		states.push_back(_normalize ? systems::normalizeDistribution(state, _mean): state);
	}

	return states;
}

// @param _maxChangeRate maximum difference between neighboring cells
// @param _expectedAvg expected average of the coefficients, if 0 it is chosen randomly from [0.1,2.5]
std::vector<System> generateSystems(size_t _numSystems, uint32_t _seed, T _maxChangeRate = 0.05, T _expectedAvg = 0.0)
{
	std::vector<System> systems;
	systems.reserve(_numSystems);

	std::default_random_engine rng(_seed);
	std::uniform_real_distribution<T> base(0.1, 2.5);
	for (size_t i = 0; i < _numSystems; ++i)
	{
		const T avg = _expectedAvg == 0.0 ? base(rng) : _expectedAvg;
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
				if (std::abs(da) > _maxChangeRate)
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

	nn::HyperParams params;
	params["name"] = std::string("heateq_net");
	params["load_net"] = false;
	params["net_type"] = std::string(typeid(NetType).name());

	// simulation
	params["time_step"] = 0.0001;
	params["hyper_sample_rate"] = USE_LOCAL_DIFFUSIFITY ? 64 : 1;

	// training
#ifdef NDEBUG
	params["train_samples"] = 128;
	params["valid_samples"] = 128;
#else
	params["train_samples"] = 1;
	params["valid_samples"] = 1;
#endif
	params["batch_size"] = 128;
	params["num_epochs"] = USE_LBFGS ? 768 : 1024;
	params["loss_p"] = 2;
	params["loss_factor"] = 100.0;
	params["loss_energy"] = 100.0;
	params["train_gpu"] = true;
	params["seed"] = static_cast<uint64_t>(7469126240319926998ull);

	// optimizer
	params["lr"] = USE_LBFGS ? 0.005 : 0.001;
	params["lr_decay"] = USE_LBFGS ? 1.0 : 0.1;
	params["lr_epoch_update"] = 512;
	params["weight_decay"] = 0.005;
	params["history_size"] = 100;

	// general
	params["depth"] = 4;
	params["bias"] = true;
	params["activation"] = nn::ActivationFn(torch::tanh);
	params["hidden_channels"] = 4;
	params["kernel_size"] = 5;
	params["residual"] = true;
	params["ext_residual"] = true;

	// cnn
	params["symmetric"] = false;

	// tcn
	params["kernel_size_temp"] = 2; // temporal dim
	params["residual_blocks"] = 3;
	params["block_size"] = 2;
	params["average"] = false;
	params["causal"] = true;
	params["interleaved"] = true;
	params["padding_mode"] = torch::nn::detail::conv_padding_mode_t(torch::kCircular);
	params["padding_mode_temp"] = torch::nn::detail::conv_padding_mode_t(torch::kZeros); // padding in temporal dim; only used when interleaved=true
	
	// best to leave these unchanged
	params["num_inputs"] = std::is_same_v<NetType, nn::Convolutional> ? 1 : NUM_INPUTS;
	params["num_outputs"] = USE_SINGLE_OUTPUT ? 1 : NUM_INPUTS;
	params["hidden_size"] = N;
	// makeNetwork uses this but does not handle the spatial dimension correctly
	params["state_size"] = USE_LOCAL_DIFFUSIFITY ? 2 : 1;
	params["num_channels"] = USE_LOCAL_DIFFUSIFITY ? 2 : 1;
	params["train_in"] = false;
	params["train_out"] = false;

	if constexpr (MODE != Mode::EVALUATE)
	{
#ifdef NDEBUG
		constexpr int numTrain = 128;
		constexpr int numValid = 32;
#else
		constexpr int numTrain = 4;
		constexpr int numValid = 2;
#endif
		auto trainingStates = generateStates(heatEq, numTrain, 0x612FF6AEu, 0.0, MEAN, STD_DEV, true);
		auto validStates = generateStates(heatEq, numValid, 0x195A4Cu, 0.0, MEAN, STD_DEV, true);
		auto warmupSteps = std::vector<size_t>{ 0, 64, 384, 256, 16, 4, 128, 2, 128, 64, 384, 256, 16, 1024, 0, 0 };

		using Integrator = std::conditional_t<USE_LOCAL_DIFFUSIFITY,
			SuperSampleIntegrator,
			systems::discretization::AnalyticHeatEq<T, N>>;

		std::vector<System> trainSystems;
		std::vector<System> validSystems;
		if constexpr (USE_LOCAL_DIFFUSIFITY)
		{
			trainSystems = generateSystems(trainingStates.size(), 0x6341241u);
			validSystems = generateSystems(validStates.size()-2, 0xBE0691u);
			std::array<T, N> heatCoefs{};
			heatCoefs.fill(0.1);
			validSystems.emplace_back(heatCoefs);
			heatCoefs.fill(3.0);
			validSystems.emplace_back(heatCoefs);
		}
		else
		{
			trainSystems.push_back(heatEq);
			validSystems.push_back(heatEq);
		}
		
		if (!torch::cuda::is_available())
			params["train_gpu"] = false;
		else if (params.get<bool>("train_gpu", true))
			std::cout << "Cuda is available. Training on GPU." << "\n";

		nn::TrainNetwork<NetType, System, Integrator, nn::MakeTensor_t<NetType>, OutputMaker, USE_WRAPPER> trainNetwork(
			trainSystems, validSystems, trainingStates, validStates, warmupSteps);

		if constexpr (MODE == Mode::TRAIN_MULTI)
		{
			constexpr int numSeeds = 8;
			std::mt19937_64 rng;
			std::vector<nn::ExtAny> seeds(numSeeds);
			std::generate(seeds.begin(), seeds.end(), rng);

			nn::GridSearchOptimizer hyperOptimizer(trainNetwork,
				{	{"kernel_size", {3, 5, 7}},
					{"hidden_channels", {2,4,6}},
				//	{"residual", {false, true}},
				//	{"bias", {false, true}},
					{"depth", {3,4,5}},
				//	{"lr", {0.02, 0.025, 0.03}},
				//	{"lr", {0.015, 0.01, 0.005}},
				//	{"lr_decay", {0.995, 0.994, 0.993}},
				//	{"amsgrad", {false, true}},
				//	{"lr_decay", {0.25, 0.1}},
				//	{"weight_decay", {0.01, 0.001, 0.0}},
				//	{"loss_energy", {10.0, 100.0, 1000.0}},
				//	{"padding_mode_temp", {torch::kZeros, torch::kCircular, torch::kReflect}}
				//	{"num_epochs", {2048}},
				//	{ "momentum", {0.5, 0.6, 0.7} },
				//	{ "dampening", {0.5, 0.4, 0.3} },
				//	{ "average", {false, true}},
				//	{"causal", {false, true}},
				//	{"activation", {nn::ActivationFn(torch::tanh), nn::ActivationFn(nn::elu), nn::ActivationFn(torch::relu)}}
					{"seed", seeds}
				//	{"seed", {7469126240319926998ull, 17462938647148434322ull}},
				}, params);

			// convolutions are already parallel in Torch so there is little benefit to
			// running multiple trainings at once
			hyperOptimizer.run(1);
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
		SuperSampleIntegrator superSampleFiniteDifs(system, timeStep, state, 64);

		auto net = nn::load<NetType, USE_WRAPPER>(params);

		eval::EvalOptions options;
		options.numShortTermSteps = 256;
		options.numLongTermRuns = 8;
		options.numLongTermSteps = 1024;
		options.mseAvgWindow = 1;
		options.downSampleRate = 4;
		options.printHeader = false;
		options.relativeError = true;

		if constexpr (USE_LOCAL_DIFFUSIFITY)
		{
			// magnitude
			std::vector<System> systems;
			std::array<T, N> heatCoefs{};

			heatCoefs.fill(0.1);
			systems.emplace_back(heatCoefs);
			heatCoefs.fill(0.5);
			systems.emplace_back(heatCoefs);
			heatCoefs.fill(1.0);
			systems.emplace_back(heatCoefs);
			heatCoefs.fill(2.0);
			systems.emplace_back(heatCoefs);
			heatCoefs.fill(2.5);
			systems.emplace_back(heatCoefs);
			heatCoefs.fill(3.0);
			systems.emplace_back(heatCoefs);
			// roughness
			systems.emplace_back(0.0);
			systems.push_back(generateSystems(1, 0xE312A41, 0.05, 1.0)[0]); // standard training sample with smooth changes
			systems.push_back(generateSystems(1, 0xFBB4F, 0.1, 1.0)[0]);
			systems.push_back(generateSystems(1, 0xFBB4F, 0.25, 1.0)[0]); 
			systems.push_back(generateSystems(1, 0xBB4F0101, 0.5, 1.0)[0]);

			systems.emplace_back(0.0);
			
			std::vector<State> states(systems.size(), state);

			// different states
			heatCoefs.fill(1.0);
			for (auto& state : validStates)
			{
				systems.push_back(heatCoefs);
				states.push_back(state);
			}

			// symmetric setup
			State symState{};
			for (size_t i = 0; i < N / 2; ++i)
			{
				symState[i] = static_cast<double>(2 * i) / N * 4.0;
				symState[N - i - 1] = symState[i];
				heatCoefs[i] = static_cast<double>(2 * i) / N * 0.5 + 0.75;
				heatCoefs[N - i - 1] = heatCoefs[i];
			}
			systems.emplace_back(heatCoefs);
			states.push_back(symState);

			for (auto& state : states )
				state = systems::normalizeDistribution(state, 0.0);

			System randSys = generateSystems(1, 0xACAB, 0.1, 1.5)[0];
			systems.emplace_back(randSys);
			states.emplace_back(*(states.end() - 3));
			
			evaluate<NUM_INPUTS>(systems, states, timeStep, options, 
				wrapNetwork<NUM_INPUTS>(net));
		}
		else
		{
			nn::Integrator<System, decltype(net), NUM_INPUTS> nnIntegrator(system, net);

			eval::evaluate(system, state, options, 
				analytic, 
				superSampleFiniteDifs, 
				finiteDiffs, 
				finiteDiffsImpl,
				nnIntegrator);
		}
	}
}
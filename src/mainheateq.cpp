#include "systems/heateq.hpp"
#include "systems/heateqsolver.hpp"
#include "nn/train.hpp"
#include "nn/mlp.hpp"
#include "nn/convolutional.hpp"
#include "nn/tcn.hpp"
#include "nn/nnintegrator.hpp"
#include "evaluation/renderer.hpp"
#include "evaluation/evaluation.hpp"
#include "evaluation/stability.hpp"
#include <random>
#include <chrono>

constexpr bool USE_WRAPPER = false;
constexpr bool USE_LOCAL_DIFFUSIFITY = true;
constexpr bool ABSOLUTE_ZERO = false;
constexpr size_t N = 32;

using T = double;
using System = systems::HeatEquation<T, N>;
using State = typename System::State;

constexpr T MEAN = ABSOLUTE_ZERO ? 128.0 : 0.0;
constexpr T STD_DEV = ABSOLUTE_ZERO ? 64.0 : 1.0;

using NetType = nn::ExtTCN;

using OutputMaker = nn::StateToTensor;
static_assert((!std::is_same_v<NetType, nn::TCN2d> && !std::is_same_v<NetType, nn::ExtTCN>) || USE_LOCAL_DIFFUSIFITY, 
	"input tensor not implemented for this config");
namespace nn {
	template<>
	struct InputMakerSelector<nn::TCN2d>
	{
		using type = systems::discretization::MakeInputHeatEq<true>;
	};

	template<>
	struct InputMakerSelector<nn::ExtTCN>
	{
		using type = systems::discretization::MakeInputHeatEq<true>;
	};

	template<>
	struct InputMakerSelector<nn::Convolutional>
	{
		using type = std::conditional_t<USE_LOCAL_DIFFUSIFITY,
			systems::discretization::MakeInputHeatEq<false>,
			nn::StateToTensor>;
	};
}

namespace disc = systems::discretization;
using SuperSampleIntegrator = disc::SuperSampleIntegrator<T, N, N * 32, N == 64 ? 1 : 2>;


// convoluted way to store the number of inputs to allow comparison of networks with different NumInputs
template<size_t NumSteps, typename Network>
struct NetWrapper
{
	NetWrapper(Network& _network) : network(_network) {}
	Network& network;
};

// typical make function to simplify creation by determining Network implicitly
template<size_t NumSteps,  typename Network>
NetWrapper<NumSteps, Network> wrapNetwork(Network& _network)
{
	return NetWrapper<NumSteps,Network>(_network);
}

template<size_t NumSteps, size_t MaxSteps>
std::array<State, NumSteps> arrayView(const std::array<State, MaxSteps>& _array)
{
	static_assert(NumSteps <= MaxSteps);

	std::array<State, NumSteps> arr{};
	constexpr size_t off = MaxSteps - NumSteps;
	for (size_t i = 0; i < NumSteps; ++i)
		arr[i] = _array[i + off];

	return arr;
}

template<typename T>
struct Unwrap
{
	constexpr static size_t numSteps = 1;
	using ContainedType = T;

	Unwrap(T& _contained) : contained(_contained) {};
	T& contained;
};

template<size_t NumSteps, typename Network>
struct Unwrap<NetWrapper<NumSteps, Network>> 
{ 
	constexpr static size_t numSteps = NumSteps; 
	using ContainedType = Network; 

	Unwrap(const NetWrapper<NumSteps, Network>& _wrapper) : contained(_wrapper.network) {};
	Network& contained;
};

template<typename Network, typename System, size_t MaxSteps>
auto makeNNIntegrator(const System& _system,
	Network&& _network,
	const std::array<State, MaxSteps>& _initialStates)
{
	using UnwrapT = Unwrap<std::remove_reference_t<Network>>;
	auto& net = UnwrapT(_network).contained;
	constexpr size_t numSteps = UnwrapT::numSteps;
	return nn::Integrator<System, typename UnwrapT::ContainedType, numSteps>(
		_system, 
		net,
		arrayView<numSteps - 1>(_initialStates));
}


template<size_t NumTimeSteps, typename... Networks>
void evaluate(
	const System& system,
	const State& _initialState,
	double _timeStep,
	const eval::ExtEvalOptions<State>& _options,
	Networks&&... _networks)
{
	disc::AnalyticHeatEq analytic(system, _timeStep, _initialState);
	disc::FiniteDifferencesExplicit<T, N, 2> finiteDiffs(system, _timeStep);
//	disc::FiniteDifferencesImplicit<T, N, 2> finiteDiffsImplicit(system, _timeStep);
	SuperSampleIntegrator superSampleFiniteDifs(system, _timeStep, _initialState, 64);

	auto [referenceIntegrator, otherIntegrator] = [&]() 
	{
		if constexpr (USE_LOCAL_DIFFUSIFITY)
			return std::make_pair(&superSampleFiniteDifs, &analytic);
		else
			return std::make_pair(&analytic, &superSampleFiniteDifs);
	}();

	// prepare initial time series
	const auto& [initialStates, initialState] = nn::computeTimeSeries<NumTimeSteps>(*referenceIntegrator, _initialState);
	referenceIntegrator->reset(system, initialState);
	otherIntegrator->reset(system, initialState);

	eval::evaluate(system,
		initialState,
		_options,
		*referenceIntegrator,
		*otherIntegrator,
		finiteDiffs,
		makeNNIntegrator(system, _networks, initialStates)...);
}

template<size_t NumTimeSteps, typename... Networks>
void evaluate(const std::vector<System>& _systems,
	const std::vector<State>& _initialStates,
	double _timeStep,
	const eval::ExtEvalOptions<State>& _options,
	Networks&&... _networks)
{
	for (size_t i = 0; i < _systems.size(); ++i)
	{
		if (_systems[i].radius() == 0.0)
		{
			std::cout << "######################################################\n";
			continue;
		}
		std::cout << "Num" << i << "\n";
		evaluate<NumTimeSteps>(_systems[i], _initialStates[i], _timeStep, _options, std::forward<Networks>(_networks)...);
	}
}

// @param _normalize shift _state so that the mean is guaranteed to be _mean.
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

template<typename... Networks>
void checkSymmetry(const System& _system, const State& _state, double _timeStep,
	const eval::EvalOptions& _options,
	Networks&&... _networks)
{

	State state{};
	System::HeatCoefficients heatCoeffs;
	for (size_t i = 0; i < N/2; ++i)
	{
	//	state[i] = _state[i];
		state[i] = static_cast<double>(i) / N * 4.0;
		state[N - i - 1] = state[i];
	//	heatCoeffs[i] = _system.heatCoefficients()[i];
		heatCoeffs[i] = static_cast<double>(i) / N * 4.0;
		heatCoeffs[N - i - 1] = heatCoeffs[i];
	}
	state = systems::normalizeDistribution(state, MEAN);

	System system(heatCoeffs);

	auto symErr = [](std::ostream& out, const State& s, const State&, double, int, int)
	{
		double err = 0.0;
		for (size_t i = 0; i < N / 2; ++i)
		{
			const double dif = s[i] - s[N - i - 1];
			err += dif * dif;
		}
		out << err;
	};

	eval::ExtEvalOptions<State> options(_options);
	options.customPrintFunctions.emplace_back(symErr);

	evaluate<NUM_INPUTS>(system, state, _timeStep, options, std::forward<Networks>(_networks)...);

/*	for (int i = 0; i < 64; ++i)
	{
		std::cout << i << "   " << symErr(state) << "\n";
		state = integrator(state);
	}*/
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
	params["name"] = std::string("yolo");
	params["load_net"] = false;

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
	params["batch_size"] = 128; // 512
	params["num_epochs"] = USE_LBFGS ? 1024 : 2048; // 768
	params["loss_p"] = 2;
	params["loss_energy"] = 1.0;
	params["train_gpu"] = true;
	params["seed"] = 7469126240319926998ull;
	params["net_type"] = std::string(typeid(NetType).name());

	// optimizer
	params["lr"] = USE_LBFGS ? 0.005 : 0.001;
	params["lr_decay"] = USE_LBFGS ? 1.0 : 0.1;
	params["lr_epoch_update"] = 768;
	params["weight_decay"] = 0.005;//0.005
	params["history_size"] = 100;

	// general
	params["depth"] = 3;
	params["bias"] = true;
	params["num_inputs"] = std::is_same_v<NetType, nn::Convolutional> ? 1 : NUM_INPUTS;
	params["num_outputs"] = USE_SINGLE_OUTPUT ? 1 : NUM_INPUTS;
	params["hidden_size"] = N;
	// makeNetwork uses this but does not handle the spatial dimension correctly
	params["state_size"] = USE_LOCAL_DIFFUSIFITY ? 2 : 1;
	params["num_channels"] = USE_LOCAL_DIFFUSIFITY ? 2 : 1;
	params["activation"] = nn::ActivationFn(torch::tanh);
	params["hidden_channels"] = 4;
	params["kernel_size"] = 5;
	params["residual"] = true;
	params["ext_residual"] = true;
	params["symmetric"] = false;

	// tcn
	params["kernel_size_temp"] = 3; // temporal dim
	params["residual_blocks"] = 3;
	params["block_size"] = 2;
	params["average"] = true;
	params["casual"] = true;
	params["interleaved"] = true;
	params["padding_mode"] = torch::nn::detail::conv_padding_mode_t(torch::kCircular);
	params["padding_mode_temp"] = torch::nn::detail::conv_padding_mode_t(torch::kZeros); // padding in temporal dim; only used when interleaved=true
	
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
	//	auto warmupSteps = std::vector<size_t>{ 0, 128, 256, 384, 512, 1024, 2048, 4096, 0, 128, 256, 0, 0, 0, 0 };

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
			constexpr int numSeeds = 4;
			std::mt19937_64 rng;
			std::vector<nn::ExtAny> seeds(numSeeds);
			std::generate(seeds.begin(), seeds.end(), rng);

			params["name"] = std::string("tcn_casual");
			nn::GridSearchOptimizer hyperOptimizer(trainNetwork,
				{//	{"kernel_size", {3, 7, 9}},
				//	{"hidden_channels", {4,6}},
				//	{"residual", {false, true}},
				//	{"bias", {false, true}},
				//	{"depth", {4,6}},
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
					{"casual", {false, true}},
				//	{"activation", {nn::ActivationFn(torch::tanh), nn::ActivationFn(nn::elu), nn::ActivationFn(torch::relu)}}
					{"seed", seeds}
				//	{"seed", {7469126240319926998ull, 17462938647148434322ull}},
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
		SuperSampleIntegrator superSampleFiniteDifs(system, timeStep, state, 64);

	//	auto net = nn::load<NetType, USE_WRAPPER>(params);
	//	nn::Integrator<System, decltype(net), NUM_INPUTS> nnIntegrator(system, net);

	//	nn::exportTensor(analytic.getGreenFn(timeStep, 63), "green.txt");
	//	eval::checkEnergy(net->layers.front(), 64);
	//	for (size_t i = 0; i < net->layers.size(); ++i)
	//		nn::exportTensor(net->layers[i]->weight, "heateq_adam2" + std::to_string(i) + ".txt");

		if constexpr (SHOW_VISUAL)
		{
			auto systems = generateSystems(36, 0x6341241);
			for (int i = 0; i < 4; ++i) 
			{
		//		disc::FiniteDifferencesExplicit finiteDiffs2(systems[i], timeStep);
				disc::SuperSampleIntegrator<T, N, N * 32> finiteDiffs2(systems[i], timeStep, trainingStates[i], 64);
				eval::HeatRenderer renderer(timeStep, N, systems[i].heatCoefficients().data(), [&, state = trainingStates[0]]() mutable
				{
					state = finiteDiffs2(state);
					return std::vector<double>(state.begin(), state.end());
				});
				renderer.run();
			}
		}

		eval::EvalOptions options;
		options.numShortTermSteps = 128;
		options.numLongTermRuns = 32;
		options.numLongTermSteps = 4096;
		options.mseAvgWindow = 0;
		options.printHeader = false;
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
	//	auto tcn = nn::load<nn::TCN2d, USE_WRAPPER>(params, "heateq32_tcn_5_3");
		if constexpr (USE_LOCAL_DIFFUSIFITY)
		{
		/*	for (auto& layer : net->layers)
			{
				int yooo = layer->weight.dim();
				std::cout << layer->weight + layer->weight.flip(2) << "\n";
				std::cout << layer->weight.flip(2) << "\n";
			}*/
			// magnitude
			std::vector<System> systems;
		/*	std::array<T, N> heatCoefs{};
			for (double i = 0.05; i < 3.0; i += 0.05)
			{
				heatCoefs.fill(i);
				systems.emplace_back(heatCoefs);
			}*/

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
				symState[i] = static_cast<double>(2 * i) / N * 2.0;
				symState[N - i - 1] = symState[i];
				heatCoefs[i] = static_cast<double>(2 * i) / N * 0.5 + 0.75;
				heatCoefs[N - i - 1] = heatCoefs[i];
			}
			systems.emplace_back(heatCoefs);
			states.push_back(symState);

			for (auto& state : states )
				state = systems::normalizeDistribution(state, 0.0);

			systems.erase(systems.begin(), systems.end() - 1);
			states.erase(states.begin(), states.end() - 1);

		//	auto net2 = nn::load<NetType, USE_WRAPPER>(params, "ext_residual_sym");
		//	checkSymmetry(generateSystems(1, 0xFBB4F, 0.1, 1.0)[0], states.back(), timeStep, options, net, net2);
		//	evaluate<NUM_INPUTS>(systems, states, timeStep, options, net, net2);
		//	return 0;

		/*	auto convBase = nn::load<nn::Convolutional, USE_WRAPPER>(params, "conv_zero_sym_base");
			auto convZero = nn::load<nn::Convolutional, USE_WRAPPER>(params, "conv_zero");
			auto convAdam = nn::load<nn::Convolutional, USE_WRAPPER>(params, "ext_residual_sym_adam");*/
		/*	auto convWd0 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "0_conv_wd");
			auto convWd0 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "0_conv_wd");
			auto convWd1 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "1_conv_wd");
			auto convSeed0 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "0_conv_seed");
			auto convSeed1 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "1_conv_seed");
			auto convSeed2 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "2_conv_seed");
			auto convSeed3 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "3_conv_seed");*/
			auto convSeed0 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "symmetric/5_conv_res_seed");
			auto convSeed1 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "symmetric/7_conv_res_seed");
			auto convSeed2 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "symmetric/8_conv_res_seed");
			auto convSeed3 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "symmetric/3_conv_sym_res_seed");
			auto convSeed4 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "energy_reg/2_1_1_conv_energy_reg");
			auto convSeed5 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "energy_reg/1_3_1_conv_energy_reg");
			auto convSeed6 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "energy_reg/2_0_1_conv_energy_reg");
			auto convNoBias1 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "bias_reg/0_3_conv_bias_reg");
			auto convNoBias2 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "bias_reg/1_4_conv_bias_reg");
			auto convReg1 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "conv_bias_reg/1_1_1_conv_bias_reg_2");
			auto convReg2 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "conv_bias_reg/0_1_1_conv_bias_reg_2");

		/*	nn::FlatConvWrapper wrapper(convSeed3);
			nn::FlatConvWrapper wrapper2(convSeed5);
			std::array<double, N> state;
			state.fill(0.0);
			for (auto& system : systems)
			{
				wrapper->constantInputs = nn::arrayToTensor(system.heatCoefficients());
			//	eval::checkEnergy(eval::computeJacobian(wrapper, nn::arrayToTensor(state)));
			//	eval::checkEnergy(eval::computeJacobian(wrapper2, nn::arrayToTensor(state)));
			}*/
			evaluate<NUM_INPUTS>(systems, states, timeStep, options,
				convSeed0,
				convSeed6,
				convReg2);

			evaluate<NUM_INPUTS>(systems, states, timeStep, options,
				convSeed0,
				convSeed3,
			//	convSeed4,
			//	convSeed5,
				convSeed6,
			//	convNoBias1,
				convNoBias2,
				convReg1,
				convReg2);

			return 0;
			// networks
		//	auto tcnInterleaved = nn::load<nn::TCN2d, USE_WRAPPER>(params, "0_tcn_interleaved_lbfgs");
		//	auto tcnInterleavedAdam = nn::load<nn::TCN2d, USE_WRAPPER>(params, "0_tcn_interleaved");
		/*	auto tcnInterleavedWd0 = nn::load<nn::TCN2d, USE_WRAPPER>(params, "0_tcn_interleaved_wd");
			auto tcnInterleavedWd1 = nn::load<nn::TCN2d, USE_WRAPPER>(params, "1_tcn_interleaved_wd");
			auto tcnInterleavedWd2 = nn::load<nn::TCN2d, USE_WRAPPER>(params, "2_tcn_interleaved_wd");
			auto tcnInterleavedWd3 = nn::load<nn::TCN2d, USE_WRAPPER>(params, "3_tcn_interleaved_wd");
			auto tcnInterleavedWd4 = nn::load<nn::TCN2d, USE_WRAPPER>(params, "4_tcn_interleaved_wd_768_epochs");
			auto tcnInterleavedWd5 = nn::load<nn::TCN2d, USE_WRAPPER>(params, "4_tcn_interleaved_wd");
			auto tcnInterleavedWd6 = nn::load<nn::TCN2d, USE_WRAPPER>(params, "0_tcn_wd_2");
			auto tcnInterleavedWd7 = nn::load<nn::TCN2d, USE_WRAPPER>(params, "1_tcn_wd_2");
			auto tcnInterleavedWd8 = nn::load<nn::TCN2d, USE_WRAPPER>(params, "0_tcn_wd_3");*/
			auto conv0 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "conv");
			auto conv1 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "conv_adam");
			auto conv2 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "0_conv");
			auto conv3 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "1_conv");
			auto conv4 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "0_conv_wd_2");
		/*	auto conv5 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "1_conv_wd_2");
			evaluate<8>(systems, states, timeStep, options, 
				wrapNetwork<8>(tcnInterleavedWd0), 
			//	wrapNetwork<8>(tcnInterleavedWd1),
			//	wrapNetwork<8>(tcnInterleavedWd2),
				wrapNetwork<8>(tcnInterleavedWd3),
				wrapNetwork<8>(tcnInterleavedWd6),
				wrapNetwork<8>(tcnInterleavedWd7),
				wrapNetwork<8>(tcnInterleavedWd8),
				wrapNetwork<1>(conv0),
				wrapNetwork<1>(conv1),
				wrapNetwork<1>(conv2),
				wrapNetwork<1>(conv3),
				wrapNetwork<1>(conv4),
				wrapNetwork<1>(conv5));*/
			auto convNew0 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "1_0_width_depth");
			auto convNew1 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "0_1_width_depth");
			auto convNew2 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "1_1_width_depth");
			auto tcnInterleavedFull = nn::load<nn::TCN2d, USE_WRAPPER>(params, "tcn_wd_2t_full");
			auto tcnInterleavedAvg = nn::load<nn::TCN2d, USE_WRAPPER>(params, "tcn_wd_2t");
		//	options.writeMSE = true;
		//	options.append = true;
		/*	nn::FlatConvWrapper wrapper(conv4);
			std::array<double, N> state;
			state.fill(0.0);
			for (auto& system : systems)
			{
				wrapper->constantInputs = nn::arrayToTensor(system.heatCoefficients());
				eval::checkEnergy(eval::computeJacobian(wrapper, nn::arrayToTensor(state)));
			}
			return 0;*/
			evaluate<8>(systems, states, timeStep, options, 
		//		wrapNetwork<NUM_INPUTS>(net),
				wrapNetwork<8>(tcnInterleavedAvg),
				wrapNetwork<8>(tcnInterleavedFull),
				wrapNetwork<1>(convNew0), 
		//		wrapNetwork<1>(convNew1),
		//		wrapNetwork<1>(convNew2),
				wrapNetwork<1>(conv4));
		}
		else
			eval::evaluate(system, state, options, analytic, superSampleFiniteDifs, finiteDiffs, finiteDiffsImpl); //nn
	}
}
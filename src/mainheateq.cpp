#include "nn/train.hpp"
#include "nn/mlp.hpp"
#include "nn/nnintegrator.hpp"
#include "evaluation/renderer.hpp"
#include "evaluation/evaluation.hpp"
#include "evaluation/stability.hpp"
#include "heateqeval.hpp"
#include <random>
#include <chrono>

using NetType = nn::ExtTCN;
static_assert((!std::is_same_v<NetType, nn::TCN2d> && !std::is_same_v<NetType, nn::ExtTCN>) || USE_LOCAL_DIFFUSIFITY, 
	"input tensor not implemented for this config");

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
	params["name"] = std::string("linear_5");
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
	params["loss_energy"] = 100.0;
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
	params["depth"] = 4;
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
	params["kernel_size_temp"] = 2; // temporal dim
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
		/*	for (auto& sys : trainSystems)
			{
				double sum = 0.0;
				for (auto d : sys.heatCoefficients())
					sum += d;
				std::cout << sum / 32 << "\n";
			}*/
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

			params["name"] = std::string("linear_kernel");
			nn::GridSearchOptimizer hyperOptimizer(trainNetwork,
				{	{"kernel_size", {5, 7, 9, 11, 13}},
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
				//	{"casual", {false, true}},
				//	{"activation", {nn::ActivationFn(torch::tanh), nn::ActivationFn(nn::elu), nn::ActivationFn(torch::relu)}}
				//	{"seed", seeds}
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

	//nn::exportTensor(analytic.getGreenFn(timeStep, 31), "green.txt");
	//	eval::checkEnergy(net->layers.front(), 64);
	//	for (size_t i = 0; i < net->layers.size(); ++i)
	//		nn::exportTensor(net->layers[i]->weight, "heateq_adam2" + std::to_string(i) + ".txt");

	/*	if constexpr (SHOW_VISUAL)
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
		}*/

		eval::EvalOptions options;
		options.numShortTermSteps = 128;
		options.numLongTermRuns = 0;
		options.numLongTermSteps = 1024;
		options.mseAvgWindow = 1;
		options.printHeader = false;
	//	options.writeGlobalError = true;
	//	options.writeMSE = true;
	//	options.append = true;
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

		/*	systems.pop_back();
			states.pop_back();
			systems.erase(systems.begin(), systems.end() - 1);
			states.erase(states.begin(), states.end() - 1);*/
		/*	systems.clear();
			states.clear();*/
		/*	systems.push_back(generateSystems(1, 0xFBB4F, 0.1, 1.0)[0]);
			states.push_back(state);*/

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

			auto convRegNew1 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "con_bias_reg_new/1_1_1_conv_bias_reg");
			auto convRegNew2 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "con_bias_reg_new/0_0_1_conv_bias_reg");
			auto convRegNew3 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "con_bias_reg_new/0_1_0_conv_bias_reg");
			auto convRegNew4 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "con_bias_reg_new/1_2_0_conv_bias_reg");
			auto convRegNew5 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "con_bias_reg_new/0_2_0_conv_bias_reg");

			auto tcnFull = nn::load<nn::ExtTCN, USE_WRAPPER>(params, "tcn/0_5_tcn");
			auto tcnAvg = nn::load<nn::ExtTCN, USE_WRAPPER>(params, "tcn/1_4_tcn");
			auto tcnAvgNoRes = nn::load<nn::ExtTCN, USE_WRAPPER>(params, "tcn/0_tcn_no_res");

			auto tcnReg0 = nn::load<nn::ExtTCN, USE_WRAPPER>(params, "tcn_reg/0_tcn_reg");
			auto tcnReg1 = nn::load<nn::ExtTCN, USE_WRAPPER>(params, "tcn_reg/1_tcn_reg");
			auto tcnReg2 = nn::load<nn::ExtTCN, USE_WRAPPER>(params, "tcn_reg/2_tcn_reg");

			auto cnnRepeat1 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "conv_repeat/1_7_conv_repeat");
			auto cnnRepeat2 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "conv_repeat/1_1_conv_repeat");

			auto cnnRepeat3 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "conv_repeat/4_conv_repeat_sym");
			auto cnnRepeat4 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "conv_repeat/7_conv_repeat_sym");
			auto cnnRepeat5 = nn::load<nn::Convolutional, USE_WRAPPER>(params, "conv_repeat/0_6_conv_repeat");
			
			// systems[8], *(states.end() - 3)
		//	makeFourierCoefsData<8>(systems.back(), states.back(), timeStep,
		//		wrapNetwork<8>(tcnAvg));
		//	makeRelativeErrorData<8>(*(systems.end() - 4), *(states.end() - 4), timeStep,
			auto validStates = generateStates(heatEq, 32, 0x195A4Cu, 0.0, MEAN, STD_DEV, true);
			auto validSystems = generateSystems(validStates.size() - 2, 0xBE0691u);
			heatCoefs.fill(0.1);
			validSystems.emplace_back(heatCoefs);
			heatCoefs.fill(3.0);
			validSystems.emplace_back(heatCoefs);
		//	makeMultiSimAvgErrorData<8>(validSystems, validStates, timeStep,
		//	makeDiffusionRoughnessData<8>(*(states.end() - 2), timeStep,
			State peakState;
			peakState.fill(0.0);
			peakState[11] = 4.0;
			peakState[22] = -4.0;
			makeStateData<8>(*(systems.end() - 4), *(states.end() - 4), timeStep, 512,
		//	makeRelativeErrorData<8>(*(systems.end() - 3), peakState, timeStep,
				cnnRepeat2
			/*	cnnRepeat5,
				cnnRepeat4,
				wrapNetwork<8>(tcnFull),
				wrapNetwork<8>(tcnAvg),
				wrapNetwork<8>(tcnAvgNoRes)*/);

		/*	makeConstDiffusionData<8>(validStates[3], timeStep,
				cnnRepeat2,
				cnnRepeat5,
				cnnRepeat4,
				wrapNetwork<8>(tcnFull),
				wrapNetwork<8>(tcnAvg),
				wrapNetwork<8>(tcnAvgNoRes));*/

			return 0;
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
		/*	evaluate<NUM_INPUTS>(systems, states, timeStep, options,
				convSeed0,
				convSeed6,
				convRegNew1,
				convRegNew2,
				convRegNew3
				convRegNew4,
				convRegNew5);
			return 0;*/
			
			evaluate<NUM_INPUTS>(systems, states, timeStep, options,
				convSeed0,
			//	cnnRepeat1,
			//	cnnRepeat2,
			//	convSeed3,
			//	convSeed4,
			//	convSeed5,
			//	convSeed6,
			//	convNoBias1,
			//	convNoBias2,
			//	convRegNew1,
			//	convRegNew2,
			//	convRegNew3,
				cnnRepeat3,
				cnnRepeat4,
				cnnRepeat5/*
				wrapNetwork<8>(tcnFull),
				wrapNetwork<8>(tcnAvg),
				wrapNetwork<8>(tcnAvgNoRes),
				wrapNetwork<8>(tcnReg0),
				wrapNetwork<8>(tcnReg1),
				wrapNetwork<8>(tcnReg2)*/);

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
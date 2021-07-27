#pragma once

#include "systems/heateq.hpp"
#include "systems/heateqsolver.hpp"
#include "nn/convolutional.hpp"
#include "nn/tcn.hpp"

#include <filesystem>
#include <thread>

constexpr bool USE_WRAPPER = false;
constexpr bool USE_LOCAL_DIFFUSIFITY = true;
constexpr bool ABSOLUTE_ZERO = false;
constexpr size_t N = 32;

using T = double;
using System = systems::HeatEquation<T, N>;
using State = typename System::State;

constexpr T MEAN = ABSOLUTE_ZERO ? 128.0 : 0.0;
constexpr T STD_DEV = ABSOLUTE_ZERO ? 64.0 : 1.0;

using OutputMaker = nn::StateToTensor;
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
template<size_t NumSteps, typename Network>
NetWrapper<NumSteps, Network> wrapNetwork(Network& _network)
{
	return NetWrapper<NumSteps, Network>(_network);
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
	const System& _system,
	const State& _initialState,
	double _timeStep,
	const eval::ExtEvalOptions<State>& _options,
	Networks&&... _networks)
{
	disc::AnalyticHeatEq analytic(_system, _timeStep, _initialState);
	disc::FiniteDifferencesExplicit<T, N, 2> finiteDiffs(_system, _timeStep);
	//	disc::FiniteDifferencesImplicit<T, N, 2> finiteDiffsImplicit(system, _timeStep);
	SuperSampleIntegrator superSampleFiniteDifs(_system, _timeStep, _initialState, 64);

	auto [referenceIntegrator, otherIntegrator] = [&]()
	{
		if constexpr (USE_LOCAL_DIFFUSIFITY)
			return std::make_pair(&superSampleFiniteDifs, &analytic);
		else
			return std::make_pair(&analytic, &superSampleFiniteDifs);
	}();

	// prepare initial time series
	const auto& [initialStates, initialState] = nn::computeTimeSeries<NumTimeSteps>(*referenceIntegrator, _initialState);
	referenceIntegrator->reset(_system, initialState);
	otherIntegrator->reset(_system, initialState);

	if constexpr (SHOW_VISUAL)
	{
		auto integrators = std::make_tuple(makeNNIntegrator(_system, _networks, initialStates)...);
		eval::HeatRenderer renderer(_timeStep, N, _system.heatCoefficients().data(), [&, state = initialState]() mutable
			{
				//	state = std::get<0>(integrators)(state);
				state = analytic(state);
				return std::vector<double>(state.begin(), state.end());
			});
		renderer.run();
	}

	eval::evaluate(_system,
		initialState,
		_options,
		*referenceIntegrator,
		*otherIntegrator,
		finiteDiffs,
		makeNNIntegrator(_system, _networks, initialStates)...);
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


template<typename... Networks>
void checkSymmetry(const System& _system, const State& _state, double _timeStep,
	const eval::EvalOptions& _options,
	Networks&&... _networks)
{

	State state{};
	System::HeatCoefficients heatCoeffs;
	for (size_t i = 0; i < N / 2; ++i)
	{
		state[i] = static_cast<double>(2 * i) / N * 4.0;
		state[N - i - 1] = state[i];
		heatCoeffs[i] = static_cast<double>(2 * i) / N * 0.5 + 0.75;
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
		out << std::sqrt(err);
	};

	eval::ExtEvalOptions<State> options(_options);
	options.customPrintFunctions.emplace_back(symErr);

	evaluate<NUM_INPUTS>(system, state, _timeStep, options, std::forward<Networks>(_networks)...);
}

template<size_t NumTimeSteps, typename... Networks>
void makeRelativeErrorData(
	const System& _system,
	const State& _initialState,
	double _timeStep,
	Networks&&... _networks)
{
	eval::ExtEvalOptions<State> options;
	options.numShortTermSteps = 1024;
	options.downSampleRate = 4;
	options.numLongTermRuns = 0;
	options.printHeader = false;
	options.writeGlobalError = true;
	options.customPrintFunctions.emplace_back(eval::RelativeError<State>());

	std::ofstream file("relative_error.txt");
	options.streams.push_back(&file);
	evaluate<NumTimeSteps>(_system, _initialState, _timeStep, options, std::forward<Networks>(_networks)...);
}

template<size_t NumTimeSteps, typename... Networks>
void makeFourierCoefsData(
	const System& _system,
	const State& _initialState,
	double _timeStep,
	Networks&&... _networks)
{
	eval::ExtEvalOptions<State> options;
	options.numShortTermSteps = 1024;
	options.numLongTermRuns = 0;
	options.downSampleRate = 4;
	options.printHeader = false;

	auto printDFT = [](std::ostream& out, const State& s, const State& ref, double err, int, int)
	{
		torch::Tensor state = nn::arrayToTensor(s);
		state = torch::fft_rfft(state);
		torch::Tensor real = torch::real(state);
		torch::Tensor imag = torch::imag(state);
		for (int64_t i = 0; i < state.size(0); ++i)
		{
			std::complex<double> c(real.index({ i }).item<double>(),
				imag.index({ i }).item<double>());
			out << std::norm(c);

			// last entry does not need a delimeter
			if(i != state.size(0)-1) out << " ";
		}
	};
	options.customPrintFunctions.emplace_back(printDFT);

	std::ofstream file("dft_coefs.txt");
	options.streams.push_back(&file);

	evaluate<NumTimeSteps>(_system, _initialState, _timeStep, options, std::forward<Networks>(_networks)...);
}

template<size_t NumTimeSteps, typename... Networks>
void makeConstDiffusionData(
	const State& _initialState,
	double _timeStep,
	Networks&&... _networks)
{
	eval::ExtEvalOptions<State> options;
	options.numShortTermSteps = 128;
	options.numLongTermRuns = 0;
	options.printHeader = false;
	options.mseAvgWindow = 1;
	options.writeMSE = true;
	options.append = true;

	const double min = 0.025;
	const double max = 3.5;
	const double stepSize = 0.025;
	std::array<T, N> heatCoefs{};

	for (double i = min; i <= max; i += stepSize)
	{
		heatCoefs.fill(i);
		options.customValueMSE = i;
		evaluate<NumTimeSteps>(System(heatCoefs), _initialState, _timeStep, options, std::forward<Networks>(_networks)...);
	}
}

template<size_t NumTimeSteps, typename... Networks>
void makeDiffusionRoughnessData(
	const State& _initialState,
	double _timeStep,
	Networks&&... _networks)
{
	eval::ExtEvalOptions<State> options;
	options.numShortTermSteps = 128;
	options.numLongTermRuns = 0;
	options.printHeader = false;
	options.mseAvgWindow = 1;
	options.writeMSE = true;
	options.append = true;

	std::vector<System> systems;
	const double min = 0.0;
	const double max = 0.5;
	const double avg = 2.0;
	const double stepSize = 0.01;
	std::array<T, N> heatCoefs{};

	for (double i = min; i <= max; i += stepSize)
	{
		heatCoefs.fill(avg);
		double sign = 1.0;
		for (size_t j = 0; j+4 < N; j+=4)
		{
			sign *= -1.0;
			heatCoefs[j+1] += i * sign;
			heatCoefs[j+2] += 2.0 * i * sign;
			heatCoefs[j + 3] += i * sign;
		}
		options.customValueMSE = i;
		evaluate<NumTimeSteps>(System(heatCoefs), _initialState, _timeStep, options, std::forward<Networks>(_networks)...);
	}
}

template<size_t NumTimeSteps, typename... Networks>
void makeMultiSimAvgErrorData(const std::vector<System>& _systems,
	const std::vector<State>& _states,
	double _timeStep, Networks&&... _networks)
{
	eval::ExtEvalOptions<State> options;
	options.numLongTermRuns = 0;
	options.numShortTermSteps = 1024;
	options.mseAvgWindow = 4;
	options.downSampleRate = 4;
	options.printHeader = false;
	options.relativeError = true;
	options.writeGlobalError = true; // needed so that averages are computed for all time steps

	eval::MultiSimulationError<State> error;
	options.customPrintFunctions.emplace_back(error.accumulateFn());
	options.streams.push_back(&eval::g_nullStream);

	for (size_t i = 0; i < _systems.size()-1; ++i)
	{
		evaluate<NumTimeSteps>(_systems[i], _states[i], _timeStep, options, _networks...);
	}
	options.customPrintFunctions.front() = error.accumulateFn(_systems.size());
	std::ofstream file("random_avg_error.txt");
	options.streams.front() = &file;
	evaluate<NumTimeSteps>(_systems.back(), _states.back(), _timeStep, options, _networks...);
}

void exportSystemState(const System& _system, const State& _state)
{
	std::ofstream file("state.txt");
	for (size_t i = 0; i < _state.size(); ++i)
	{
		file << _system.heatCoefficients()[i] << " " << _state[i] << "\n";
	}
}

template<size_t NumTimeSteps, typename... Networks>
void makeStateData(const System& _system, const State& _state, double _timeStep,
	std::vector<int> _steps, Networks&&... _networks)
{
	eval::ExtEvalOptions<State> options;
	options.numLongTermRuns = 0;
	options.mseAvgWindow = 1;
	options.numShortTermSteps = _steps.back()+1;
	options.downSampleRate = 0;
	options.printHeader = false;

	std::unordered_map<int, std::vector<State>> states;
	for (int step : _steps)
		states.emplace(step, std::vector<State>(sizeof...(Networks) + 3));

	auto printState = [&](std::ostream& out, const State& s, const State& ref, double err, 
		int _timeStep, int _integrator)
	{
		auto it = states.find(_timeStep);
		if(it != states.end())
			it->second[_integrator] = s;
	};
	options.customPrintFunctions.emplace_back(printState);
	options.streams.push_back(&eval::g_nullStream);

	evaluate<NumTimeSteps>(_system, _state, _timeStep, options, _networks...);

	std::ofstream file("state.txt");
	for (size_t i = 0; i < _state.size(); ++i)
	{
		file << _system.heatCoefficients()[i] << " ";
		for (auto& [step, integratorStates] : states)
		{
			for (auto& s : integratorStates)
				file << s[i] << " ";
		}
		file << "\n";
	}
}

template<size_t NumTimeSteps, typename... Networks>
void makeEnergyData(const System& _system, const State& _state, double _timeStep, Networks&&... _networks)
{
	constexpr int maxSamples = 256;
	eval::ExtEvalOptions<State> options;
	options.numLongTermRuns = 0;
	options.numShortTermSteps = 1024 * 1024;
	options.mseAvgWindow = 1;
	options.downSampleRate = options.numShortTermSteps / maxSamples;
	options.printHeader = false;
	options.writeEnergy = true;

	evaluate<NumTimeSteps>(_system, _state, _timeStep, options, _networks...);
}

template<size_t NumTimeSteps, typename... Networks>
void makeEnergyDataMulti(
	const State& _initialState,
	double _timeStep, Networks&&... _networks)
{
	constexpr int maxSamples = 256;
	eval::ExtEvalOptions<State> options;
	options.numLongTermRuns = 0;
	options.numShortTermSteps = 1024 * 128;
	options.mseAvgWindow = 1;
	options.downSampleRate = options.numShortTermSteps / maxSamples;
	options.printHeader = false;
	options.writeEnergy = true;

	std::vector<double> coeffs{ 0.1,0.5,1.0,1.5,2.0,2.5,3.0,3.5 };
	std::array<T, N> heatCoefs{};

	for (size_t i = 0; i < coeffs.size(); ++i)
	{
		heatCoefs.fill(coeffs[i]);
		evaluate<NumTimeSteps>(System(heatCoefs), _initialState, _timeStep, options, _networks...);
		std::filesystem::path file = "energy.txt";
		std::filesystem::rename(file, "energy_" + std::to_string(i) + ".txt");
	}
}

template<size_t NumTimeSteps, typename... Networks>
void checkZeroStability(
	const std::vector<System>& _systems,
	double _timeStep, Networks&&... _networks)
{
	eval::ExtEvalOptions<State> options;
	options.numLongTermRuns = 256;
	options.numLongTermSteps = 4;
	options.printHeader = false;

	std::vector<State> states(_systems.size(), State{});
	evaluate<NumTimeSteps>(_systems, states, _timeStep, options, _networks...);
}

template<size_t NumTimeSteps, typename... Networks>
void makeStabilityData(
	const State& _state,
	double _timeStep, const nn::Convolutional& _network)
{
	nn::FlatConvWrapper wrapper(_network, nn::FlatConvMode::Stack);
	nn::FlatConvWrapper wrapper2(_network, nn::FlatConvMode::ConstDiffusion);

	const double min = 0.025;
	const double max = 3.5;
	const double stepSize = 0.025;
	std::array<T, N> heatCoefs{};

	const torch::Tensor zeros = torch::zeros({ N }, c10::TensorOptions(c10::kDouble));
	std::ofstream file("eigs_coef.txt");

	for (double i = min; i <= max; i += stepSize)
	{
		heatCoefs.fill(i);
		const torch::Tensor coefs = nn::arrayToTensor(heatCoefs);
		wrapper2->constantInputs = coefs;
		auto eigsJ = eval::computeEigs(eval::computeJacobian(wrapper2,
			zeros));

		auto eigsM = eval::checkEnergy(eval::computeJacobian(wrapper,
			torch::cat({ coefs, zeros })), N);

		double maxEigJ = std::numeric_limits<double>::min();
		double maxEigM = std::numeric_limits<double>::min();
		for (size_t i = 0; i < N; ++i)
		{
			const double v = std::abs(eigsJ[i]);
			if (v > maxEigJ) maxEigJ = v; 
			if (eigsM[i] > maxEigM) maxEigM = eigsM[i];
		}

		file << i << " " << maxEigJ << " " << maxEigM << "\n";

	/*	//	makeEnergyDataMulti<8>(states.back(), timeStep, convRegNew3);
		//	auto eigs = eval::computeEigs(eval::toMatrix(analytic.getGreenFn(timeStep), 32));
		nn::FlatConvWrapper wrapper(convRegNew3, nn::FlatConvMode::Stack);
		nn::FlatConvWrapper wrapper2(convRegNew3, nn::FlatConvMode::ConstDiffusion);
		heatCoefs.fill(0.0);
		wrapper2->constantInputs = nn::arrayToTensor(heatCoefs);
		auto eigs = eval::computeEigs(eval::computeJacobian(wrapper2,
			torch::zeros({ N }, c10::TensorOptions(c10::kDouble))));
		//	auto eigs = eval::checkEnergy(eval::computeJacobian(wrapper, 
		//		torch::zeros({ 2 * N }, c10::TensorOptions(c10::kDouble))), N);
		for (auto eig : eigs)
			std::cout << std::abs(eig) << "\n";
		heatCoefs.fill(0.0);
		state = *(states.end() - 2);
		//	state.fill(1.0);
		//	state.fill(1.0);
		//	for (size_t i = 0; i < 32; ++i)
		//		state[i] = i % 2 ? -1.0 : 1.0;
		makeEnergyData<1>({ System(heatCoefs) }, state, timeStep, convRegNew3);
		return 0;*/
	}
}

template<typename It, typename Fn>
void runMultiThreaded(It _begin, It _end, Fn _fn, size_t _numThreads = 1)
{
	if (_numThreads == 1)
		_fn(_begin, _end);
	else
	{
		std::vector<std::thread> threads;
		threads.reserve(_numThreads - 1);

		using DistanceType = decltype(_end - _begin);
		const DistanceType n = static_cast<DistanceType>(_numThreads);
		const DistanceType rows = (_end - _begin) / n;
		for (DistanceType i = 0; i < n - 1; ++i)
			threads.emplace_back(_fn, i * rows, (i + 1) * rows);
		_fn((n - 1) * rows, _end);

		for (auto& thread : threads)
			thread.join();
	}
}

void checkSteadyState(const std::vector<System>& _systems, 
	const std::vector<State>& _states,
	double _timestep)
{
	std::mutex printMutex;
	auto execute = [&](size_t begin, size_t end)
	{
		for (size_t i = begin; i < end; ++i)
		{
			const System& system = _systems[i];
			State state = _states[i];
			SuperSampleIntegrator superSampleFiniteDifs(system, _timestep, state, 64);
			const double initialAvg = systems::average(state);
			double energy = system.energy(state);
			double oldEnergy;
			do {
				for (size_t i = 0; i < 128; ++i) {
					state = superSampleFiniteDifs(state);
				}
				oldEnergy = energy;
				energy = system.energy(state);
			} while (std::abs(oldEnergy - energy) / oldEnergy > 0.000001);

			std::unique_lock<std::mutex> lock(printMutex);
			std::cout << initialAvg << " " 
				<< systems::average(state) << " "
				<< energy << "\n";
		}
	};

	runMultiThreaded(static_cast<size_t>(0), _systems.size(), execute, 8);
}


// just copied from mainheateq so it will need some changes before compiling
/*void checkEigsLinear()
{
	std::vector<double> realParts;
	std::vector<double> othOnes;
	for (size_t i = 0; i < 15; ++i)
	{
		std::cout << i << "\n";
		std::string name = "linear_cnn/";
		if (i < 7) name += std::to_string(i) + "_linear_kernel";
		else name += std::to_string(i - 7) + "_linear_kernel_2";
		auto linear1 = nn::load<nn::Convolutional, USE_WRAPPER>(params, name);
		auto eigs = eval::computeEigs(eval::toMatrix(linear1->layers.back()->weight, 32));

		double max = 0.0;
		for (const auto& eig : eigs)
		{
			if (eig.imag() > max)
			{
				max = eig.imag();
			}
			if (eig.real() > 1.0)
				othOnes.push_back(eig.real());
		}
		realParts.push_back(max);
	}

	std::cout << std::setprecision(std::numeric_limits<long double>::digits10 - 8);
	for (size_t i = 0; i < realParts.size(); ++i)
	{
		std::cout << i * 2 + 3 << " " << realParts[i] << "\n";
	}

	for (double d : othOnes)
		std::cout << d << "\n";
}*/

void processSizeData(const nn::HyperParams& _params, const std::string& _file) 
{
	std::ifstream file(_file);
	std::ofstream outputFile("cnn_weights.txt");
	while (!file.eof())
	{
		std::string name;
		double loss = 0.0;

		file >> name >> loss;
		auto net = nn::load<nn::Convolutional, USE_WRAPPER>(_params, 
			"cnn_scale_size/" + name);

		outputFile /*<< name << " "*/ << loss << " " << nn::countParams(net) << "\n";
	}
}

void toyExample()
{
	constexpr size_t EXAMPLE_N = 4;
	std::array<T, EXAMPLE_N> heatCoefs{ 1.1,1.1,1.0,1.0 };
	systems::HeatEquation<T, EXAMPLE_N> example(heatCoefs);
	disc::FiniteDifferencesExplicit<T, EXAMPLE_N, 1> fdm(example, 0.1);
	std::array<T, EXAMPLE_N> exState{ 1.0,1.0,-1.0,-1.0 };
	for (size_t i = 0; i < 300; ++i)
	{
		exState = fdm(exState);
		std::cout << systems::average(exState) << " " << example.energy(exState) << "\n";
	}
}
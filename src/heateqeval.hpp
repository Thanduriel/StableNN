#pragma once

#include "systems/heateq.hpp"
#include "systems/heateqsolver.hpp"
#include "nn/convolutional.hpp"
#include "nn/tcn.hpp"

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
	double _timeStep, Networks&... _networks)
{
	eval::ExtEvalOptions<State> options;
	options.numLongTermRuns = 0;
	options.numShortTermSteps = 1024;
	options.mseAvgWindow = 4;
	options.downSampleRate = 4;
	options.printHeader = false;
	options.relativeError = true;
	options.writeGlobalError = true; // needed so that averages are computed for all time steps

	class NullBuffer : public std::streambuf
	{
	public:
		int overflow(int c) { return c; }
	};

	NullBuffer nullBuffer;
	std::ostream nullStream(&nullBuffer);

	eval::MultiSimulationError<State> error;
	options.customPrintFunctions.emplace_back(error.accumulateFn());
	options.streams.push_back(&nullStream);

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
	int _steps, Networks&... _networks)
{
	eval::ExtEvalOptions<State> options;
	options.numLongTermRuns = 0;
	options.numShortTermSteps = _steps;
	options.mseAvgWindow = 0;
	options.downSampleRate = 0;
	options.printHeader = false;

	std::vector<State> states;

	auto printState = [&](std::ostream& out, const State& s, const State& ref, double err, 
		int _timeStep, int _integrator)
	{
		if (_integrator >= states.size())
			states.resize(_integrator + 1);
		states[_integrator] = s;
	};

	options.customPrintFunctions.emplace_back(printState);
	evaluate<NumTimeSteps>(_system, _state, _timeStep, options, _networks...);

	std::ofstream file("state.txt");
	for (size_t i = 0; i < _state.size(); ++i)
	{
		file << _system.heatCoefficients()[i] << " ";
		for(auto& s : states)
			file << s[i] << " ";
		file << "\n";
	}
}

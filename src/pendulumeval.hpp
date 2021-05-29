#include "systems/pendulum.hpp"
#include "systems/odesolver.hpp"
#include "systems/rungekutta.hpp"
#include "nn/hyperparam.hpp"
#include "nn/nnintegrator.hpp"
#include "evaluation/evaluation.hpp"
#include "evaluation/renderer.hpp"
#include "evaluation/stability.hpp"
#include "evaluation/lipschitz.hpp"
#include "evaluation/asymptotic.hpp"
#include "constants.hpp"
#include "defs.hpp"

#include <iostream>
#include <type_traits>
#include <filesystem>

using System = systems::Pendulum<double>;
using State = typename System::State;

// simulation related
constexpr int HYPER_SAMPLE_RATE = 128;
constexpr bool USE_SIMPLE_SYSTEM = true;

namespace discret = systems::discretization;
using LeapFrog = discret::ODEIntegrator<System, discret::LeapFrog>;

template<size_t NumTimeSteps, typename... Networks>
void evaluate(
	const System& system,
	const State& _initialState,
	double _timeStep,
	eval::EvalOptions _options,
	Networks&... _networks)
{
	LeapFrog leapFrog(system, _timeStep);
	discret::ODEIntegrator<System, discret::ForwardEuler> forwardEuler(system, _timeStep);
	discret::ODEIntegrator<System, discret::RungeKutta<discret::RK2_heun>> rk2(system, _timeStep);
	discret::ODEIntegrator<System, discret::RungeKutta<discret::RK3_ssp>> rk3(system, _timeStep);
	discret::ODEIntegrator<System, discret::RungeKutta<discret::RK4>> rk4(system, _timeStep);
//	discret::ODEIntegrator<System, StaticResNet> resNetBench(system, _timeStep);

	auto referenceIntegrate = [&](const State& _state)
	{
		LeapFrog forward(system, _timeStep / HYPER_SAMPLE_RATE);
		auto state = _state;
		for (int i = 0; i < HYPER_SAMPLE_RATE; ++i)
			state = forward(state);
		return state;
	};

	/*	eval::PendulumRenderer renderer(0.05);
		renderer.addIntegrator([drawIntegrator = rk2, state = _initialState]() mutable
		{
			state = drawIntegrator(state);
			return state.position;
		});
		renderer.run();*/

		// prepare initial time series
	const auto& [initialStates, initialState] = nn::computeTimeSeries<NumTimeSteps>(referenceIntegrate, _initialState);

	if constexpr (SHOW_VISUAL)
	{
		// copy integrator because nn::Integrator may have an internal state
		auto integrators = std::make_tuple(nn::Integrator<System, Networks, NumTimeSteps>(system, _networks, initialStates)...);
		auto visState = initialState;
		// warmup to see long term behavior
		for (int i = 0; i < 10000; ++i)
		{
			visState = std::get<0>(integrators)(visState);
		}

		eval::PendulumRenderer renderer(_timeStep);
		renderer.addIntegrator([drawIntegrator = std::get<0>(integrators), state = visState]() mutable
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

	eval::evaluate(system,
		initialState,
		_options,
		referenceIntegrate,
		leapFrog,
	//	rk2,
		rk3,
		rk4,
	//	resNetBench,
		nn::Integrator<System, Networks, NumTimeSteps>(system, _networks, initialStates)...);
}

template<size_t NumTimeSteps, typename... Networks>
void evaluate(const System& _system,
	const std::vector<State>& _initialStates,
	double _timeStep,
	const eval::EvalOptions& _options,
	Networks&... _networks)
{
	for (const State& state : _initialStates)
		evaluate<NumTimeSteps>(_system, state, _timeStep, _options, _networks...);
}

template<size_t NumTimeSteps, typename... Networks>
void makeEnergyErrorData(const System& _system, double _timeStep, int _numSteps, int _avgWindow, Networks&... _networks)
{
	eval::EvalOptions options;
	options.writeMSE = true;
	options.numLongTermRuns = 0;
	options.numShortTermSteps = _numSteps;
	options.mseAvgWindow = _avgWindow;
	options.append = true;
	options.addInitialStateMSE = true;

	constexpr int numStates = 256;
	std::vector<State> states;
	states.reserve(numStates);
	for (int i = 0; i < numStates; ++i)
		states.push_back({ static_cast<double>(i) / numStates * PI, 0.0 });

	evaluate<NumTimeSteps>(_system, states, _timeStep, options, _networks...);
}

template<size_t NumTimeSteps, typename... Networks>
void makeGlobalErrorData(const System& _system, const State& _state, double _timeStep, int _numSteps, int _avgWindow, Networks&... _networks)
{
	eval::EvalOptions options;
	options.numLongTermRuns = 0;
	options.writeGlobalError = true;
	options.numShortTermSteps = 10000;
	options.mseAvgWindow = 64;
	options.downSampleRate = 32; // 1.02765 

	evaluate<NumTimeSteps>(_system, _state, _timeStep, options, _networks...);
}

template<typename NetType>
void makeStableFrequencyData(const System& system, const nn::HyperParams& params)
{
	//{ 0.1, 0.05, 0.025, 0.01, 0.005 };
	std::vector<double> timeSteps = /*{ 0.1, 0.05, 0.025, 0.01, 0.005 };*/{ 0.05, 0.049, 0.048, 0.047, 0.046, 0.045 };
	std::vector<std::pair<std::string, double>> names;
	for (int j = 4; j < 6; ++j)
		for (int i = 0; i < 4; ++i)
		{
			const std::string name = std::to_string(i) + "_" + std::to_string(j) + "_closeFreq.pt";
			if (std::filesystem::exists(name))
				names.emplace_back(name, timeSteps[j]);
		}

	std::mutex outputMutex;
	auto computeFrequencies = [&](size_t begin, size_t end)
	{
		for (size_t i = begin; i < end; ++i)
		{
			const auto& [name, timeStep] = names[i];
			auto param = params;
			param["time_step"] = timeStep;
			auto othNet = nn::makeNetwork<NetType, true>(param);
			torch::load(othNet, name);

			nn::Integrator<System, decltype(othNet), NUM_INPUTS> integrator(system, othNet);
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
	for (size_t i = 0; i < numThreads - 1; ++i)
	{
		threads.emplace_back(computeFrequencies, i * numTasks, (i + 1) * numTasks);
	}
	computeFrequencies((numThreads - 1) * numTasks, names.size());
	for (auto& t : threads) t.join();
}

// evaluate the Jacobian on a grid
template<typename Network, typename Fn, typename RefNet = int>
void makeJacobianData(
	Network& _network,
	const State& _min,
	const State& _max,
	const systems::Vec<int, 2>& _steps,
	Fn _discard = [](const State&) {return false; },
	RefNet _refNet = 0)
{
	using namespace torch;
	std::ofstream file1("spectral_radius.txt");
	std::ofstream file2("determinant.txt");

	auto computeEigs = [](auto& network, const Tensor& state)
	{
		double max = 0.0;
		std::complex<double> det = 1.0;

		const Tensor J = eval::computeJacobian(network, state);
		const auto eigenvalues = eval::computeEigs(J);
		for (auto eig : eigenvalues)
		{
			const double abs = std::abs(eig);
			if (abs > max) max = abs;
			det *= eig;
		}

		return std::pair<double, double>(max, det.real());
	};

	const double posStep = (_max.position - _min.position) / _steps[0];
	const double velStep = (_max.velocity - _min.velocity) / _steps[1];
	for (double v = _min.velocity; v <= _max.velocity; v += velStep)
	{
		for (double p = _min.position; p <= _max.position; p += posStep)
		{
			double spectralRad = 0.0;
			double determinant = 1.0;

			// values outside are marked as infinity so that pgfplots can skip them
			if (_discard(State{ p,v }))
			{
				spectralRad = std::numeric_limits<double>::infinity();
				determinant = std::numeric_limits<double>::infinity();
			}
			else
			{
				const Tensor z = tensor({ p, v }, c10::TensorOptions(c10::kDouble));
				auto [max, det] = computeEigs(_network, z);
				spectralRad = max;
				determinant = det;

				if constexpr(!std::is_same_v<RefNet,int>)
				{
					auto [max2, det2] = computeEigs(_refNet, z);
					spectralRad = spectralRad-max2;
				//	determinant = determinant-det2;
				}
			}
			file1 << p << " " << v << " " << spectralRad << "\n";
			file2 << p << " " << v << " " << determinant << "\n";
		}
		file1 << "\n";
		file2 << "\n";
	}
}

void makeVerletPeriodErrorData(const System& _system, double _timeStep, int _refSampleRate);

template<size_t NumTimeSteps, typename Integrator>
void makeAsymptoticEnergyData(const System& _system, Integrator& _integrator)
{
	std::vector<State> states;
	const auto& [attractors, repellers] = eval::findAttractors(_system, _integrator, true);
	for (const auto& [lower, upper] : repellers)
	{
		states.push_back(State{ lower,0.0 });
		states.push_back(State{ upper,0.0 });
	}
	/*	states.push_back({ 0.0 ,0.0 });
		states.push_back({ 1.13307, 0.0});  // ResNet
		states.push_back({ 1.1309, 0.0 });
		states.push_back({ 2.67286, 0.0 });
		states.push_back({ 2.67373, 0.0 });*/
		/*	states.push_back({ 1.02086, 0.0 }); // Hamiltonian
			states.push_back({ 1.02163, 0.0 });*/
			/*	states.push_back({ 0.0, 0.0 });
				states.push_back({ 0.00076699, 0.0 });
				states.push_back({ 3.12, 0.0 }); // AntiSym
				states.push_back({ 3.13, 0.0 });*/

	constexpr int numSteps = 150000;
	constexpr int downSample = numSteps / 256;

	std::ofstream file("asymptotic.txt");
	for (int i = 0; i < numSteps; ++i)
	{
		if (i % downSample == 0)
		{
			file << i << " ";
			for (auto& state : states)
			{
				file << _system.energy(state) << " ";
			}
			file << "\n";
		}
		for (auto& state : states)
		{
			if (_system.energy(state) > 8.0) continue;
			state = _integrator(state);
		}
	}
}

template<size_t NumTimeSteps, typename Network>
void makeSinglePhasePeriodData(const System& _system, const State& _state, double _timeStep, Network& _network, bool _attractor = false)
{
	State state = _state;
	if (_attractor) {
		nn::Integrator<System, Network, NUM_INPUTS> integrator(_system, _network);
		for (int i = 0; i < 100000; ++i)
			state = integrator(state);
	}
	LeapFrog leapFrog(_system, _timeStep);
	const double period = eval::computePeriodLength(state, leapFrog, 8);

	eval::EvalOptions options;
	options.numShortTermSteps = std::ceil(period)+1;
	options.writeState = true;
	options.downSampleRate = 1;
	evaluate<NUM_INPUTS>(_system, state, _timeStep, options, _network);
}

template<typename NetType, typename TrainFn>
void makeLipschitzData(const nn::HyperParams& _params, TrainFn trainNetwork)
{
	std::vector<double> times{ 0.125/2/*, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125*/ };
	std::vector<nn::HyperParams> bestNets;
	nn::HyperParams params = _params;

	
	for (double time : times)
	{
		params["time"] = time;
		params["name"] = std::string("hamiltonian_lipschitz_") + std::to_string(static_cast<int>(time * 100));
		nn::GridSearchOptimizer hyperOptimizer(trainNetwork, {
		//	{"seed", {93784130ul, 167089119616745849ull, 23423223442167775ull, 168488165347327969ull, 116696928402573611ull, 17932827895858725ull, 51338360522333466ull, 100818424363624464ull}}
			{"seed", {9378341130ul, 16708911996216745849ull, 2342493223442167775ull, 16848810653347327969ull, 11664969248402573611ull, 1799302827895858725ull, 5137385360522333466ull, 10088183424363624464ull}}
			}, params);
		bestNets.emplace_back(hyperOptimizer.run(8));
	}

	std::ofstream file("lipschitz.txt");
	for (auto& params : bestNets)
	{
		auto net = nn::load<NetType, true>(params);
		file << *params.get<double>("time") << " " 
			<< eval::lipschitz(net) << " " 
			<< *params.get<double>("validation_loss") << "\n";
	}
}

template<typename... Integrators>
void makeRuntimeData(const System& _system, double _timeStep, int _numSteps, Integrators&&... _integrators)
{
	LeapFrog leapFrog(_system, _timeStep);
	discret::ODEIntegrator<System, discret::RungeKutta<discret::RK2_midpoint>> rk2(_system, _timeStep);
	discret::ODEIntegrator<System, discret::RungeKutta<discret::RK2_heun>> rk2Heun(_system, _timeStep);
	discret::ODEIntegrator<System, discret::RungeKutta<discret::RK3_ssp>> rk3(_system, _timeStep);
	discret::ODEIntegrator<System, discret::RungeKutta<discret::RK4>> rk4(_system, _timeStep);

	State state{ PI * 3.0 / 4.0, 0.0 };

	eval::measureRunTimes(state, _numSteps, 1, leapFrog, rk2, rk2Heun, rk3, rk4, std::forward<Integrators>(_integrators)...);
/*	std::cout << "Verlet ";
	e += eval::measureRunTime(state, leapFrog, _numSteps);
	std::cout << "RK2-Midpoint ";
	e += eval::measureRunTime(state, rk2, _numSteps);
	std::cout << "RK2-Heun ";
	e += eval::measureRunTime(state, rk2Heun, _numSteps);
	std::cout << "RK3-SSP ";
	e += eval::measureRunTime(state, rk3, _numSteps);
	std::cout << "RK4 ";
	e += eval::measureRunTime(state, rk4, _numSteps);*/
}

namespace nn {
	// implementation of the verlet integrator with torch
	class VerletPendulumImpl : public torch::nn::Cloneable<VerletPendulumImpl>
	{
	public:
		VerletPendulumImpl(double _timeStep, int _hyperSampleRate)
			: timeStep(_timeStep / _hyperSampleRate), hyperSampleRate(_hyperSampleRate) {}

		void reset() override {};

		torch::Tensor forward(const torch::Tensor& _input);

	private:
		double timeStep;
		int hyperSampleRate;
	};

	TORCH_MODULE(VerletPendulum);
}

// hard coded network for performance tests
class StaticResNet
{
	using Weights = systems::Matrix<double, 4, 4>;

	constexpr static double Scale = 0.7071067811865475;
	constexpr static systems::Matrix<double, 4, 2> P1 = { Scale, 0.0, Scale, 0.0, 0.0, Scale, 0.0, Scale };
	constexpr static systems::Matrix<double, 2, 4> P2 = { Scale, Scale, 0.0, 0.0, 0.0, 0.0, Scale, Scale };

	template<typename T, std::size_t N, std::size_t M>
	static systems::Matrix<T, N, M> activation(const systems::Matrix<T, N, M>& m)
	{
		systems::Matrix<T, N, M> result;
		for (size_t i = 0; i < N * M; ++i)
			result[i] = std::tanh(m[i]);
		return result;
	}
public:
	template<typename System, typename State, typename T>
	State operator()(const System& _system, const State& _state, T _dt) const
	{
		using namespace systems;
		constexpr Weights W1{ -0.508018846876296170655962,0.107195858590926756948036,0.183575896011169387156414,-0.220792921795074287283356,
							   0.186493842800449061147816,-0.076538653175044796261872,0.168479382717763415122647,0.028031348908101650502234,
							  -0.409280792172205476475710,0.494945093695868865157905,-0.232615918539272520382255,0.429333954431309339216938,
							   0.065267004569061662366813,-0.014283261696808003879400,-0.249809727259388825171271,0.097494080476053818218318};
		constexpr Weights W2{  0.454166705393713399097066,-0.008810583036501021259035,-0.961461566805976497462893,1.446059018019847819402912,
							  -0.232844401616651514030920,0.108096189547219098670006,-1.496503608041853228272089,2.406068693990545792615876,
							  -0.482626986160281379323322,-0.325771970126074505991198,0.063837590603751936946253,-0.146885245430029509616787,
							   3.639562459815796291451306,-1.986652898162546421190200,1.066143521275290551031389,-0.899613786404198201296367 };

		constexpr auto W1P = W1 * P1;

		Vec<double, 4> s = P1 * static_cast<Vec<double, 2>>(_state);
	//	s += activation(W1 * s);
		s += activation(W1P * static_cast<Vec<double, 2>>(_state));
		s += activation(W2 * s);

		return P2 * s;
	}
};

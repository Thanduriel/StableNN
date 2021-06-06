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
	const eval::ExtEvalOptions<State>& _options,
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
	const eval::ExtEvalOptions<State>& _options,
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
void makeMultiSimAvgErrorData(const System& _system, double _timeStep, int _numSteps, int _avgWindow, double _maxDisplacement, Networks&... _networks)
{
	eval::ExtEvalOptions<State> options;
	options.numLongTermRuns = 0;
	options.numShortTermSteps = _numSteps;
	options.mseAvgWindow = _avgWindow;
	options.downSampleRate = _avgWindow;
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

	constexpr int numStates = 256;
	for (int i = 0; i < numStates; ++i)
	{
		State state{ static_cast<double>(i) / numStates * _maxDisplacement, 0.0 };
		evaluate<NumTimeSteps>(_system, state, _timeStep, options, _networks...);
	}
	State state{ _maxDisplacement, 0.0 };
	options.customPrintFunctions.front() = error.accumulateFn(numStates);
	std::ofstream file("large_nets_avg_error.txt");
	options.streams.front() = &file;
	evaluate<NumTimeSteps>(_system, state, _timeStep, options, _networks...);
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
// @return the largest singular value or Lipschitz constant
template<typename Network, typename Fn, typename RefNet = int>
double makeJacobianData(
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
	double lipschitz = 0.0;

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
		const double norm = torch::linalg_norm(J, 2).item<double>();

		return std::tuple<double, double,double>(max, det.real(), norm);
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
				auto [max, det, norm] = computeEigs(_network, z);
				spectralRad = max;
				determinant = det;
				lipschitz = std::max(norm, lipschitz);

				if constexpr(!std::is_same_v<RefNet,int>)
				{
					auto [max2, det2, norm2] = computeEigs(_refNet, z);
					spectralRad = spectralRad-max2;
				//	determinant = determinant-det2;
				}
			}
			file1 << p << " " << v << " " << spectralRad << "\n";
			file2 << p << " " << v << " " << determinant << "\n";
		}
		file1 << "\n";
		file2 << "\n";

		return lipschitz;
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
	std::vector<double> times{ 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125 };
	std::vector<nn::HyperParams> bestNets;
	nn::HyperParams params = _params;

	
	for (double time : times)
	{
		params["time"] = time;
		params["name"] = std::string("resnet_lipschitz_") + std::to_string(static_cast<int>(time * 100));
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
		auto restrictNone = [&](const State& _state) { return false; };
		const double lipschitz = makeJacobianData(net, { -PI, -2.0 }, { PI, 2.0 }, { 64, 64 }, restrictNone);

		file << *params.get<double>("time") << " " 
			<< eval::lipschitz(net) << " " 
			<< *params.get<double>("validation_loss") / 100.0  << " "
			<< lipschitz << "\n";
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

	State state{ PI * _numSteps / (_numSteps*2.0 + 0.1), 0.0 };
	
	eval::measureRunTimes(state, _numSteps, 16, leapFrog, rk2, rk2Heun, rk3, rk4, std::forward<Integrators>(_integrators)...);
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
	// resnet 2_4
//	constexpr static systems::Matrix<double, 4, 2> P1 = { Scale, 0.0, Scale, 0.0, 0.0, Scale, 0.0, Scale };
//	constexpr static systems::Matrix<double, 2, 4> P2 = { Scale, Scale, 0.0, 0.0, 0.0, 0.0, Scale, Scale };

/*	constexpr static Weights W1{-0.508018846876296170655962,0.107195858590926756948036,0.183575896011169387156414,-0.220792921795074287283356,
							   0.186493842800449061147816,-0.076538653175044796261872,0.168479382717763415122647,0.028031348908101650502234,
							  -0.409280792172205476475710,0.494945093695868865157905,-0.232615918539272520382255,0.429333954431309339216938,
							   0.065267004569061662366813,-0.014283261696808003879400,-0.249809727259388825171271,0.097494080476053818218318 };
	constexpr static Weights W2{ 0.454166705393713399097066,-0.008810583036501021259035,-0.961461566805976497462893,1.446059018019847819402912,
						  -0.232844401616651514030920,0.108096189547219098670006,-1.496503608041853228272089,2.406068693990545792615876,
						  -0.482626986160281379323322,-0.325771970126074505991198,0.063837590603751936946253,-0.146885245430029509616787,
						   3.639562459815796291451306,-1.986652898162546421190200,1.066143521275290551031389,-0.899613786404198201296367 };*/


	// resnet 2_4l
	constexpr static systems::Matrix<double, 4, 2> P1 = { -0.276016163174906736799130,0.106984489706263327657432,
														0.031506647275956534137720,0.006743082718290418532681,
														0.867802947996441753630847,0.309057724764775831882702,
														-0.417404581425632248414814,0.572263152048785861403246 };
	constexpr static systems::Matrix<double, 2, 4> P2 = { 0.142414217079043448066500,0.728503847786360902993863,1.027910803365936542874692,-0.214379030536262032979877,
														-0.876360648810033615596637,0.039893567956600074764228,0.670753780954308598261093,1.630884939493464358406527 };
	constexpr static Weights W1{ -0.440272808479328092712279,0.495252317450778578589166,0.177339694700984207287320,0.011092765179805487429920,
								0.314252806758209435322726,-0.309790369993776237844685,0.030392172139348565956807,-0.002169501855058145710353,
								-0.185947810213055508832269,-0.034046161058796077514277,-0.326913180945591907988756,0.146908374793296048199664,
								0.247912775473209945342745,0.107669146011740712864935,-0.242761439510104259920098,-0.030216519726731880246540 };
	constexpr static Weights W2{ -0.543089577156085079323589,0.272454070281403337716597,-0.102155774765578674645461,0.155648653006292214673323,
								0.116105313659017175820232,-0.053197674134154410152675,0.260565793777785048579432,-0.104603156235785019201323,
								-0.289935847748416408720118,0.117590340813998606162016,0.224710782036964817898195,-0.038829429497712807106691,
								-1.202008059870305567429227,0.033887533827453651669170,0.419769785233023273729458,0.132707973412622304287822};

	// resNet4_4l
/*	constexpr static systems::Matrix<double, 4, 2> P1 = { 0.930085139508391334217663, 0.381562472861024071235647,
														-0.532564589192990212040968, 0.158812182022318643115355,
														0.024498081030134816193922, 0.511869739506339627155285,
														1.686631718721591344234412, 0.139243527220944185440743};
	constexpr static systems::Matrix<double, 2, 4> P2 = { 0.215708339069449056557559, -0.190125057103728151153277, 0.267629347366463044011198, 0.364093730973153117957963,
														0.326069164106546793835406, 0.792710760428187222181862, 1.484411097070396312602725, 0.043029654552344288875876 };
	constexpr static Weights W1{ 0.078569947808206350159388, 0.078343127108518775814083, 0.029422206209004372196025, -0.278034552581518878966449,
								0.393074935217622600802656, 0.015800637285029367484768, -0.195073131347550721148565, 0.021919515897412023691659,
								-0.315509083844561444287535, 0.585865394467079636520168, -0.116008409369331183524920, 0.160731052102362548250625,
								0.186887719552804226763598, -1.021191605934270718591961, 0.358126510072897530356784, -0.088812173703245900213687};
	constexpr static Weights W2{ -0.327405014332456001380223, -0.585289601021454997109572, 0.395179958829846911250172, 0.158336868094620164537645,
								-0.247946483168673875718824, 0.230237736863188047209405, -0.024424210550320780283018, 0.113129502201324352861356,
								-0.156754826491491583428939, 0.435590142606508856637504, -0.140395825141914121214626, 0.261564277202662842647385,
								-0.192503610989824769372802, -0.505019474437582327475127, 0.387618657659834942030841, 0.299364177614767468238455};
	constexpr static Weights W3{ -0.555698252098201339599370, -0.152882957055631318876721, 0.446944864900726657186425, -0.155050398755677137918596,
								-0.199024603487359452724803, -0.781981650675289796659229, 0.502938702531510073434617, -0.154081307982009751977870,
								0.714365665404671945637460, 0.076320955207277191290061, -0.381941624731765128064609, -0.187666225418731097418501,
								-0.274863247388275167004679, -0.666266358539869596455674, 0.484244578555084403959086, -0.248631483575848938327724};
	constexpr static Weights W4{ 0.063006892587312818276857, -0.045374952468251579518199, -0.126225326323202657885503, 0.262967302364347677023915,
								0.105396932092074832598705, -0.608301939473625652432531, 0.206860585172888133964619, 0.215927431711197020947068,
								0.180774679944726401892297, 0.055141473447872672852821, -0.143877860776390170682859, -0.246391839971395104624818,
								-0.057869787801368692548021, -0.504518233255785042423724, 0.241013699806580050655214, 0.284574383622904047985713};*/


	constexpr static auto W1P = W1 * P1;

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

		Vec<double, 4> s = P1 * static_cast<Vec<double, 2>>(_state);
	//	s += activation(W1 * s);
		s += activation(W1P * static_cast<Vec<double, 2>>(_state));
		s += activation(W2 * s);
	//	s += activation(W3 * s);
	//	s += activation(W4 * s);

		return P2 * s;
	}
};

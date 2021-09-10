#include "systems/harmonicoscillator.hpp"
#include "systems/odesolver.hpp"
#include "systems/rungekutta.hpp"
#include "evaluation/evaluation.hpp"

using T = double;
using System = systems::HarmonicOscillator<T>;
using State = typename System::State;

constexpr int HYPER_SAMPLE_RATE = 32;

struct AnalyticOscillator
{
	AnalyticOscillator(const System& _system, T _dt, const State& _initialState)
		: m_system(_system),
		m_dt(_dt)
	{
		const T ratio = m_system.damping() / 2;
		m_frequency = m_system.frequency();
		const T temp = _initialState[1] + _initialState[0] * ratio;

		m_amplitude = std::sqrt(_initialState[0] * _initialState[0]
			+ temp * temp / (m_frequency * m_frequency));
		m_phaseShift = std::tanh(temp / (m_frequency * _initialState[0]));

		std::cout << m_amplitude << " " << m_phaseShift << "\n";
	}

	State operator()(const State& _state)
	{
		m_t += m_dt;
		const T ratio = m_system.damping() / 2;
		const T decay = std::exp(-ratio * m_t);
		const T x = m_frequency * m_t - m_phaseShift;
		return { m_amplitude * decay * std::cos(x),
			m_amplitude * decay * (-m_frequency * std::sin(x) - ratio * std::cos(x))};
	}
private:
	System m_system;
	T m_dt;
	T m_t = 0.0;
	T m_frequency;
	T m_amplitude;
	T m_phaseShift;
};

void evaluate(
	const System& system,
	const State& _initialState,
	double _timeStep,
	int _numSteps)
{
	namespace discret = systems::discretization;
//	discret::ODEIntegrator<System, discret::LeapFrog> leapFrog(system, _timeStep);
	discret::ODEIntegrator<System, discret::ForwardEuler> forwardEuler(system, _timeStep);
	discret::ODEIntegrator<System, discret::RungeKutta<discret::RK2_heun>> rk2(system, _timeStep);
	discret::ODEIntegrator<System, discret::RungeKutta<discret::RK3_ssp>> rk3(system, _timeStep);
	discret::ODEIntegrator<System, discret::RungeKutta<discret::RK4>> rk4(system, _timeStep);
	AnalyticOscillator analytic(system, _timeStep, _initialState);

	auto referenceIntegrate = [&](const State& _state)
	{
		discret::ODEIntegrator<System, discret::RungeKutta<discret::RK4>> forward(system, _timeStep / HYPER_SAMPLE_RATE);
		auto state = _state;
		for (int i = 0; i < HYPER_SAMPLE_RATE; ++i)
			state = forward(state);
		return state;
	};

	eval::EvalOptions options;
	options.numShortTermSteps = _numSteps;
	options.numLongTermRuns = 0;
	options.writeState = true;
	options.mseAvgWindow = 4;

	eval::evaluate(system,
		_initialState,
		options,
		referenceIntegrate,
	//	leapFrog,
		forwardEuler,
		rk2,
		rk3,
		rk4,
		analytic);
}

int main()
{
	System system(1.0, 0.25);
	State initialState{ 1.0, -0.125 };

	std::cout << system.toInitialState(1.0, 0.0) << std::endl;

	evaluate(system, initialState, 0.25, 512);

	return 0;
}
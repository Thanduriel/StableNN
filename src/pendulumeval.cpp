#include "pendulumeval.hpp"

void makeVerletPeriodErrorData(const System& _system, double _timeStep, int _refSampleRate)
{
	LeapFrog integrator(_system, _timeStep);
	LeapFrog reference(_system, _timeStep / _refSampleRate);
	auto referenceIntegrate = [&](const State& _state)
	{
		auto state = _state;
		for (int i = 0; i < _refSampleRate; ++i)
			state = reference(state);
		return state;
	};
	LeapFrog integrator2(_system, _timeStep / 2);
	auto integrate2 = [&](const State& _state)
	{
		auto state = _state;
		for (int i = 0; i < 2; ++i)
			state = integrator2(state);
		return state;
	};
	LeapFrog integrator3(_system, _timeStep / 4);
	auto integrate3 = [&](const State& _state)
	{
		auto state = _state;
		for (int i = 0; i < 4; ++i)
			state = integrator3(state);
		return state;
	};

	std::ofstream file("frequency.txt");

	constexpr int numStates = 256;
	constexpr int periods = 4096;
	for (int i = 0; i < numStates; ++i)
	{
		State state{ static_cast<double>(i) / numStates * PI, 0.0 };
		double ref = eval::computePeriodLength(state, referenceIntegrate, periods);
		double verlet = eval::computePeriodLength(state, integrator, periods);
		double verlet2 = eval::computePeriodLength(state, integrate2, periods);
		double verlet3 = eval::computePeriodLength(state, integrate3, periods);
		file << state.position << " "
			<< _system.energy(state) << " "
			<< ref << " "
			<< std::abs(ref - verlet) << " "
			<< std::abs(ref - verlet2) << " "
			<< std::abs(ref - verlet3) << "\n";
	}
}

namespace nn {
	torch::Tensor VerletPendulumImpl::forward(const torch::Tensor& _input)
	{
		using namespace torch::indexing;

		const int64_t halfSize = 1;
		torch::Tensor p = _input.dim() == 2 ? _input.index({ "...", Slice(0, halfSize) }) : _input.index({ Slice(0, halfSize) });
		torch::Tensor v = _input.dim() == 2 ? _input.index({ "...", Slice(halfSize) }) : _input.index({ Slice(halfSize) });
		for (int i = 0; i < hyperSampleRate; ++i)
		{
			const auto a0 = -torch::sin(p);
			p = p + v * timeStep + 0.5 * timeStep * timeStep * a0;

			const auto a1 = -torch::sin(p);
			v = v + 0.5 * timeStep * (a0 + a1);
		}

		return torch::cat({ p,v }, _input.dim() - 1);
	}
}
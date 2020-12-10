#pragma once

#include <iostream>
#include <fstream>
#include <array>

namespace eval {

	namespace details {
		template<size_t Ind, typename StateArray, typename Integrator, typename... Integrators>
		void evaluateStep(StateArray& _states, Integrator& _integrator, Integrators&... _integrators)
		{
			_states[Ind] = _integrator(_states[Ind]);

			if constexpr (static_cast<bool>(sizeof...(Integrators)))
				evaluateStep<Ind + 1>(_states, _integrators...);
		}

	}

	// Simulates the given system with different integrators to observe energy over time.
	template<typename System, typename State, typename... Integrators>
	void evaluate(const System& _system, const State& _initialState, Integrators&&... _integrators)
	{
		constexpr size_t numIntegrators = sizeof...(_integrators);

		std::array<State, numIntegrators> currentState;
		std::array<double, numIntegrators> cumulativeError{};
		for (auto& state : currentState) state = _initialState;

	//	std::ofstream spaceTimeFile("spacetime.txt");
	//	std::ofstream energyFile("energy.txt");

		std::cout << "initial energy: " << _system.energy(_initialState) << std::endl;
		std::cout.precision(5);
		std::cout << std::fixed;

		// short term
		constexpr int numSteps = 128;
		for (int i = 0; i < numSteps; ++i)
		{
			details::evaluateStep<0>(currentState, _integrators...);

			for (size_t j = 0; j < numIntegrators; ++j)
			{
				const auto& state = currentState[j];
 				const double dx = state.position - currentState[0].position;
				const double dv = state.velocity - currentState[0].velocity;
				cumulativeError[j] += std::sqrt(dx * dx + dv * dv);
			//	std::cout << std::sqrt(dx * dx + dv * dv) << " ";
			//	energyFile << _system.energy(state) << " ";
			//	spaceTimeFile << std::fmod(state.position, 2.0 * 3.14159) << ", ";
			}
		//	std::cout << "\n";
		//	energyFile << "\n";
		//	spaceTimeFile << "\n";
		}
	//	spaceTimeFile.flush();
	//	energyFile.flush();

		std::cout << "mse============================================" << "\n";
		for (double err : cumulativeError)
		{
			std::cout << err / numSteps << " ";
		}

		// long term energy behavior
		std::cout << "\nlongterm=======================================" << "\n";
		for (int i = 0; i < 4; ++i)
		{
			for (int j = 0; j < 4096; ++j)
			{
				details::evaluateStep<0>(currentState, _integrators...);
			}

			for (auto& state : currentState)
				std::cout << _system.energy(state) << ", ";
			std::cout << "\n";
		}
	}

}
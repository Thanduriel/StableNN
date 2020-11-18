#pragma once

#include <iostream>
#include <fstream>
#include <array>

namespace eval {

	// Simulates the given system with different integrators to observe energy over time.
	template<typename System, typename State, typename... Integrators>
	void evaluate(const System& _system, const State& _initialState, Integrators&&... _integrators)
	{
		constexpr size_t numIntegrators = sizeof...(_integrators);

		std::array<System::State, numIntegrators> currentState;
		for (auto& state : currentState) state = _initialState;

		std::ofstream spaceTimeFile("spacetime.txt");

		std::cout << "initial energy: " << _system.energy(_initialState) << std::endl;
		// short term
		for (int i = 0; i < 256; ++i)
		{
			details::evaluateStep<0>(currentState, _integrators...);

			for (auto& state : currentState)
			{
				std::cout << _system.energy(state) << ", ";
				spaceTimeFile << std::fmod(state.position, 2.0 * 3.1415) << ", ";
			}
			std::cout << "\n";
			spaceTimeFile << "\n";
		}
		spaceTimeFile.flush();

		// long term energy behavior
		std::cout << "longterm=======================================" << "\n";
		for (int i = 0; i < 8; ++i)
		{
			for (int j = 0; j < 2048; ++j)
			{
				details::evaluateStep<0>(currentState, _integrators...);
			}

			for (auto& state : currentState)
				std::cout << _system.energy(state) << ", ";
			std::cout << "\n";
		}
	}

	namespace details {
		template<size_t Ind, typename StateArray, typename Integrator, typename... Integrators>
		void evaluateStep(StateArray& _states, Integrator& _integrator, Integrators&... _integrators)
		{
			_states[Ind] = _integrator(_states[Ind]);
			evaluateStep<Ind + 1>(_states, _integrators...);
		}

		template<size_t Ind, typename StateArray>
		void evaluateStep(StateArray& _states)
		{
		}
	}

}
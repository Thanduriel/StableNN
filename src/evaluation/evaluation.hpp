#pragma once

#include <iostream>
#include <fstream>
#include <array>
#include <vector>
#include "../systems/state.hpp"

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

	struct EvalOptions
	{
		bool writeEnergy = false;
		bool writeState = false;
		bool writeMSE = false;
		bool append = false; // append to file instead of overwriting it for all write options
		int mseAvgWindow = 0; // take average of a number of previous time-steps; if 0 use numShortTermSteps
		int numShortTermSteps = 256;
		int numLongTermSteps = 4096;
		int numLongTermRuns = 4;
	};

	// Simulates the given system with different integrators to observe energy over time.
	// For error computations the first integrator is used as reference.
	template<typename System, typename State, typename... Integrators>
	void evaluate(
		const System& _system,
		const State& _initialState,
		const EvalOptions& _options,
		Integrators&&... _integrators)
	{
		constexpr size_t numIntegrators = sizeof...(_integrators);

		using StateArray = std::array<State, numIntegrators>;
		StateArray currentState;
		std::vector<StateArray> stateLog;
		std::array<double, numIntegrators> cumulativeError{};
		for (auto& state : currentState) state = _initialState;

		stateLog.push_back(currentState);

		std::cout << "===============================================\n";
		const auto initialEnergy = _system.energy(_initialState);
		std::cout << "initial state:  " << _initialState << "\n";
		std::cout << "initial energy: " << initialEnergy << std::endl;

		// short term simulation
		stateLog.reserve(_options.numShortTermSteps + 1);
		for (int i = 0; i < _options.numShortTermSteps; ++i)
		{
			details::evaluateStep<0>(currentState, _integrators...);
			stateLog.push_back(currentState);
		}

		const int avgWindow = _options.mseAvgWindow ? _options.mseAvgWindow : _options.numShortTermSteps;
		for (int i = _options.numShortTermSteps - avgWindow; i < _options.numShortTermSteps; ++i)
		{
			for (size_t j = 0; j < numIntegrators; ++j)
			{
				const auto& state = stateLog[i][j];
				double err = 0.0;
				for (size_t k = 0; k < state.size(); ++k)
				{
					const double d = state[k] - stateLog[i][0][k];
					err += d * d;
				}
				cumulativeError[j] += std::sqrt(err);
			}
		}

		auto evalSteps = [&](std::ostream& out, auto printFn)
		{
			for (int i = 0; i < _options.numShortTermSteps; ++i)
			{
				for (size_t j = 0; j < numIntegrators; ++j)
				{
					const auto& state = stateLog[i][j];
					printFn(out, state);
				}
				out << "\n";
			}
			out.flush();
		};

		if (_options.writeEnergy)
		{
			std::ofstream energyFile("energy.txt");

			evalSteps(energyFile, [&](std::ostream& out, const State& state)
				{
					out << _system.energy(state) << " ";
				});
		}

		if (_options.writeState)
		{
			std::ofstream spaceTimeFile("spacetime.txt", _options.append ? std::ios::app : std::ios::out);

			evalSteps(spaceTimeFile, [&](std::ostream& out, const State& state)
				{
					out << state << ", ";
				});
		}

		if (_options.writeMSE)
		{
			std::ofstream mseFile("mse.txt", _options.append ? std::ios::app : std::ios::out);
			mseFile << initialEnergy << ", ";
			for (double err : cumulativeError)
			{
				mseFile << err / avgWindow << ", ";
			}
			mseFile << "\n";
		}

		std::cout.precision(5);
		std::cout << std::fixed;

		std::cout << "mse -------------------------------------------\n";
		for (double err : cumulativeError)
		{
			std::cout << err / avgWindow << ", ";
		}
		std::cout << "\n";

		if (_options.numLongTermRuns)
		{
			// long term energy behavior
			std::cout << "longterm --------------------------------------" << "\n";
			for (int i = 0; i < _options.numLongTermRuns; ++i)
			{
				for (int j = 0; j < _options.numLongTermSteps; ++j)
				{
					details::evaluateStep<0>(currentState, _integrators...);
				}

				for (auto& state : currentState)
					std::cout << _system.energy(state) << ", ";
				std::cout << "\n";
			}
		}
	}
}
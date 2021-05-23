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

	template<typename State>
	auto l2Error(const State& v, const State& w)
	{
		assert(v.size() == w.size());

		using T = std::remove_cv_t<std::remove_reference_t<decltype(v[0])>>;

		T sum = 0.0;
		for (size_t k = 0; k < v.size(); ++k)
		{
			const T d = v[k] - w[k];
			sum += d * d;
		}
		return std::sqrt(sum);
	}

	struct EvalOptions
	{
		bool printHeader = true;
		bool writeEnergy = false;
		bool writeState = false;
		bool writeGlobalError = false;
		bool writeMSE = false;
		bool addInitialStateMSE = false; // print state in addition to energy in mse file
		bool append = false; // append to file instead of overwriting it for all write options
		int downSampleRate = 1; // write only every n-th entry; adds timestep as first column
		int mseAvgWindow = 0; // take average of a number of previous time-steps; if 0 use numShortTermSteps
		int numShortTermSteps = 256;
		int numLongTermSteps = 4096;
		int numLongTermRuns = 4;
	};

	template<typename State>
	struct ExtEvalOptions : public EvalOptions
	{
		ExtEvalOptions(const EvalOptions& _options) : EvalOptions(_options) {}

		using PrintFn = void(std::ostream&, const State&, const State&, int, int);
		std::vector<std::function<PrintFn>> customPrintFunctions;
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
		evaluate(_system, _initialState, ExtEvalOptions<State>(_options), std::forward<Integrators>(_integrators)...);
	}

	template<typename System, typename State, typename... Integrators>
	void evaluate(
		const System& _system,
		const State& _initialState,
		const ExtEvalOptions<State>& _options,
		Integrators&&... _integrators)
	{
		constexpr size_t numIntegrators = sizeof...(_integrators);

		using StateArray = std::array<State, numIntegrators>;
		StateArray currentState;
		std::vector<StateArray> stateLog;

		for (auto& state : currentState) state = _initialState;

		stateLog.push_back(currentState);

		std::cout << "===============================================\n";
		const auto initialEnergy = _system.energy(_initialState);
		if (_options.printHeader)
		{
			std::cout << "initial state:  " << _initialState << "\n";
			std::cout << "initial energy: " << initialEnergy << std::endl;
		}

		// short term simulation
		const int numShortTermSteps = _options.numShortTermSteps;
		stateLog.reserve(numShortTermSteps + 1);
		for (int i = 0; i < numShortTermSteps; ++i)
		{
			details::evaluateStep<0>(currentState, _integrators...);
			stateLog.push_back(currentState);
		}

		const size_t avgWindow = static_cast<size_t>(_options.mseAvgWindow ? _options.mseAvgWindow : numShortTermSteps);
		const size_t start = static_cast<size_t>(_options.writeGlobalError ? 0 : numShortTermSteps - avgWindow);
		std::vector<std::array<double, numIntegrators>> globalError;
		globalError.reserve(numShortTermSteps - start);

		for (int i = start; i < numShortTermSteps; ++i)
		{
			globalError.push_back({});
			auto& err = globalError.back();
			const auto& refState = stateLog[i][0];
			for (size_t j = 0; j < numIntegrators; ++j)
			{
				const auto& state = stateLog[i][j];
				err[j] = l2Error(state, refState);
			}
		}

		// compute averages
		std::vector<std::array<double, numIntegrators>> globalErrorAvg;
		globalErrorAvg.reserve(globalError.size());
		std::array<double, numIntegrators> currentAvg{};

		for (size_t i = 0; i < globalError.size(); ++i)
		{
			globalErrorAvg.push_back({});
			auto& avgErr = globalErrorAvg.back();
			for (size_t j = 0; j < numIntegrators; ++j)
			{
				currentAvg[j] += globalError[i][j];
				if (i >= avgWindow)
				{
					currentAvg[j] -= globalError[i - avgWindow][j];
					avgErr[j] = currentAvg[j] / avgWindow;
				}
				else
				{
					avgErr[j] = currentAvg[j] / (i+1);
				}
			}
		}
		auto& cumulativeError = globalErrorAvg.back();

		const int downSampleRate = _options.downSampleRate ? _options.downSampleRate : 1;
		auto evalSteps = [&](std::ostream& out, auto printFn, const std::string& delim = " ")
		{
			for (int i = 0; i < numShortTermSteps; i += downSampleRate)
			{
				if (downSampleRate > 1)
					out << i << delim;

				const State& refState = stateLog[i][0];
				for (size_t j = 0; j < numIntegrators; ++j)
				{
					const State& state = stateLog[i][j];
					printFn(out, state, refState, i, static_cast<int>(j));
					out << delim;
				}
				out << "\n";
			}
			out.flush();
		};
		auto fileOptions = _options.append ? std::ios::app : std::ios::out;

		if (_options.writeEnergy)
		{
			std::ofstream energyFile("energy.txt");

			evalSteps(energyFile, [&](std::ostream& out, const State& state, const State&, int, int)
				{
					out << _system.energy(state);
				});
		}

		if (_options.writeState)
		{
			std::ofstream spaceTimeFile("spacetime.txt", fileOptions);

			evalSteps(spaceTimeFile, [&](std::ostream& out, const State& state, const State&, int, int)
				{
					out << state;
				}, ", ");
		}

		if (_options.writeGlobalError)
		{
			std::ofstream file("globalerror.txt", fileOptions);

			evalSteps(file, [&](std::ostream& out, const State& state, const State& refState, 
				int step, int integrator)
				{
					out << globalErrorAvg[step][integrator];
				});
		}

		if (_options.writeMSE)
		{
			std::ofstream mseFile("mse.txt", fileOptions);
			mseFile << initialEnergy << ", ";
			if (_options.addInitialStateMSE)
				mseFile << _initialState << ", ";
			for (double err : cumulativeError)
			{
				mseFile << err << ", ";
			}
			mseFile << "\n";
		}

		for (auto& printFn : _options.customPrintFunctions)
		{
			evalSteps(std::cout, printFn);
		}

		std::cout.precision(5);
		std::cout << std::fixed;

		std::cout << "mse -------------------------------------------\n";
		for (double err : cumulativeError)
		{
			std::cout << err << ", ";
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
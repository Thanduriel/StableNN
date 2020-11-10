#pragma once

#include "nn/dataset.hpp"

template<typename System, typename Integrator>
class DataGenerator
{
	using SysState = typename System::State;
public:
	DataGenerator(const System& _system, const Integrator& _integrator) 
		: m_system(_system), m_integrator(_integrator)
	{
	}

	// @param _numSamples Number of samples to generate for each initial state.
	// @param _downSampleRate Number of steps to simulate for each sample.
	// @param _numInputSteps Number of steps which are combined to one sample for inputs in a timeseries.
	nn::Dataset generate(const std::vector<SysState>& _initialStates,
		size_t _numSamples = 256,
		size_t _downSampleRate = 1,
		int64_t _numInputSteps = 1)
	{
		assert(_initialStates.size() >= 1);

		const int64_t stateSize = sizeof(SysState) / sizeof(System::ValueT);
		const int64_t inputSize = _numInputSteps * stateSize;

		torch::Tensor inputs;
		torch::Tensor outputs;

		for (auto& state : _initialStates)
		{
			auto results = runSimulation(state, (_numSamples+_numInputSteps) * _downSampleRate, _downSampleRate);
			std::vector<SysState> timeSeries;
			timeSeries.reserve(_numSamples * _numInputSteps);
			for(size_t i = 0; i < _numSamples; ++i)
				for (int64_t j = 0; j < _numInputSteps; ++j)
				{
					timeSeries.push_back(results[i + j]);
				}

			const int64_t size = static_cast<int64_t>(results.size()) - _numInputSteps;
			torch::Tensor in = torch::from_blob(timeSeries.data(), { size, _numInputSteps * stateSize }, c10::TensorOptions(c10::ScalarType::Double));
			torch::Tensor out = torch::from_blob(results.data() + _numInputSteps, { size, stateSize }, c10::TensorOptions(c10::ScalarType::Double));

			if (!inputs.defined())
			{
				inputs = in.clone();
				outputs = out.clone();
			}
			else
			{
				inputs = torch::cat({ inputs, in });
				outputs = torch::cat({ outputs, out });
			}
		}

		return { inputs, outputs };
	}
private:
	auto runSimulation( const SysState& _initialState,
		int _steps,
		int _subSteps)
	{
		SysState state{ _initialState };
		std::vector<System::State> results;
		results.reserve(_steps);
		results.push_back(state);

		for (int i = 0; i < _steps; ++i)
		{
			state = m_integrator(state);
			if (i % _subSteps == 0)
				results.push_back(state);
		}

		return results;
	}

	System m_system;
	Integrator m_integrator;
};
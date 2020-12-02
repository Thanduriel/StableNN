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
	// @param _numInputSteps Number of steps which are combined to one sample for inputs in a time series.
	// @param _numOutputSteps Number of steps which are combined to one sample for outputs in a time series.
	//						  A value of zero means that _numInputSteps are given.
	nn::Dataset generate(const std::vector<SysState>& _initialStates,
		int64_t _numSamples = 256,
		int64_t _downSampleRate = 1,
		int64_t _numInputSteps = 1,
		bool _useSingleOutput = true,
		int64_t _inOutShift = 1) const
	{
		assert(_initialStates.size() >= 1);
		if (_numInputSteps == 1) _useSingleOutput = true;

		const int64_t stateSize = sizeof(SysState) / sizeof(typename System::ValueT);
		const int64_t inputSize = _numInputSteps * stateSize;
		const int64_t outputSize = _useSingleOutput ? stateSize : inputSize;

		// steps required without considering _numInputSteps
		const int64_t samplesReq = _numSamples + _inOutShift;

		torch::Tensor inputs;
		torch::Tensor outputs;

		for (auto& state : _initialStates)
		{
			auto results = runSimulation(state, samplesReq + _numInputSteps - 1, _downSampleRate);
			std::vector<SysState> timeSeries;

			const int64_t samplesForSeries = _useSingleOutput ? _numSamples : samplesReq;
			timeSeries.reserve(samplesForSeries * _numInputSteps);
			for(int64_t i = 0; i < samplesForSeries; ++i)
				for (int64_t j = 0; j < _numInputSteps; ++j)
				{
					timeSeries.push_back(results[i + j]);
				}

			const int64_t size = _numSamples;
			torch::Tensor in = torch::from_blob(timeSeries.data(), { size, inputSize }, c10::TensorOptions(c10::ScalarType::Double));
			torch::Tensor out = torch::from_blob(
				_useSingleOutput ? (results.data() + _numInputSteps + _inOutShift - 1) : (timeSeries.data() + _numInputSteps * _inOutShift),
				{ size, outputSize }, 
				c10::TensorOptions(c10::ScalarType::Double));

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
		size_t _steps,
		size_t _subSteps) const
	{
		SysState state{ _initialState };
		std::vector<typename System::State> results;
		results.reserve(_steps);
		results.push_back(state);

		const size_t computeSteps = _steps * _subSteps;
		// start at 1 because we already have the initial state
		for (size_t i = 1; i < computeSteps; ++i)
		{
			state = m_integrator(state);
			if (i % _subSteps == 0)
				results.push_back(state);
		}

	//	for (auto& s : results)
	//		s.velocity *= 0.1 * 0.5 * 0.5;

		return results;
	}

	System m_system;
	Integrator m_integrator;
};
#pragma once

#include "helpers.hpp"
#include "nn/dataset.hpp"
#include "nn/utils.hpp"
#include "systems/state.hpp"

template<typename System, typename Integrator, typename InputMaker = nn::StateToTensor, typename OutputMaker = nn::StateToTensor>
class DataGenerator
{
	using SysState = typename System::State;
public:
	DataGenerator(const System& _system, const Integrator& _integrator) 
		: DataGenerator({ _system }, _integrator)
	{
	}

	DataGenerator(std::vector<System> _systems, const Integrator& _integrator)
		: m_systems(std::move(_systems)),
		m_integrator(_integrator),
		m_tensorOptions(c10::CppTypeToScalarType<typename System::ValueT>())
	{
	}

	// @param _numSamples Number of samples to generate for each initial state.
	// @param _downSampleRate Number of steps to simulate for each sample.
	// @param _numInputSteps Number of steps which are combined to one sample for inputs in a time series.
	// @param _numOutputSteps Number of steps which are combined to one sample for outputs in a time series.
	//						  A value of zero means that _numInputSteps are given.
	// @param _warmup Number of steps to simulate before starting to record the actual samples. 
	//				  If less elements are provided then _initialStates, they are reused.
	nn::Dataset generate(const std::vector<SysState>& _initialStates,
		int64_t _numSamples = 256,
		int64_t _downSampleRate = 1,
		int64_t _numInputSteps = 1,
		bool _useSingleOutput = true,
		int64_t _inOutShift = 1,
		const std::vector<size_t> _warmup = {}) const
	{
		assert(_initialStates.size() >= 1);
		if (_numInputSteps == 1) _useSingleOutput = true;

		// steps required without considering _numInputSteps
		const int64_t samplesReq = _numSamples + _inOutShift;

		torch::Tensor inputs;
		torch::Tensor outputs;

		for(size_t k = 0; k < _initialStates.size(); ++k)
		{
			const SysState& state = _initialStates[k];
			const System& system = m_systems[k % m_systems.size()];
			auto results = runSimulation(system, state, samplesReq + _numInputSteps - 1, _downSampleRate, 
				_warmup.empty() ? static_cast<size_t>(0) : _warmup[k % _warmup.size()]);
			std::vector<SysState> timeSeries;

			const int64_t samplesForSeries = _useSingleOutput ? _numSamples : samplesReq;
			timeSeries.reserve(samplesForSeries * _numInputSteps);
			for(int64_t i = 0; i < samplesForSeries; ++i)
				for (int64_t j = 0; j < _numInputSteps; ++j)
				{
					timeSeries.push_back(results[i + j]);
				}

			InputMaker stateToTensor;
			const int64_t batchSize = _numSamples;
			torch::Tensor in = stateToTensor(system, timeSeries.data(), timeSeries.size(), batchSize, m_tensorOptions);
			
			OutputMaker outputMaker;
			SysState* outData = _useSingleOutput ? (results.data() + _numInputSteps + _inOutShift - 1) 
				: (timeSeries.data() + _numInputSteps * _inOutShift);
			torch::Tensor out = outputMaker(system, outData, _useSingleOutput ? batchSize : batchSize * _numInputSteps, batchSize, m_tensorOptions);
		/*	torch::from_blob(
				_useSingleOutput ? (results.data() + _numInputSteps + _inOutShift - 1) : (timeSeries.data() + _numInputSteps * _inOutShift),
				{ numSamples, outputSize },
				m_tensorOptions);*/

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
	auto runSimulation( const System& _system,
		const SysState& _initialState,
		size_t _steps,
		size_t _subSteps,
		size_t _warmup) const
	{
		SysState state{ _initialState };
		std::vector<SysState> results;
		results.reserve(_steps);

		// a statefull integrator with non const operator() needs to be recreated
		Integrator integrator = [this, &_system, &_initialState]() 
		{
			if constexpr (!is_callable<const Integrator, SysState>::value)
				return Integrator(m_integrator, _system, _initialState);
			else 
				return Integrator(_system, m_integrator.deltaTime());
		}();

		for (size_t i = 0; i < _warmup; ++i)
			state = integrator(state);

		const size_t computeSteps = _steps * _subSteps;
		results.push_back(state);
		// start at 1 because we already have the initial state
		for (size_t i = 1; i < computeSteps; ++i)
		{
			state = integrator(state);
			if (i % _subSteps == 0)
				results.push_back(state);
		}

		return results;
	}

	std::vector<System> m_systems;
	Integrator m_integrator;
	c10::TensorOptions m_tensorOptions;
};
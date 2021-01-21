#pragma once

#include "../systems/state.hpp"
#include "utils.hpp"
#include <c10/core/ScalarType.h>
#include <array>

namespace nn{

	template<typename System, typename Network, size_t NumStates = 1, typename InputMaker = StateToTensor>
	class Integrator
	{
		using State = typename System::State;
		using T = typename System::ValueT;
	public:
		Integrator(const System& _system, Network& _network, const std::array<State, NumStates-1>& _initialState = {})
			: m_system(_system),
			m_network(_network),
			m_options(c10::CppTypeToScalarType<T>())
		{
			_network->to(c10::CppTypeToScalarType<T>());
			for (size_t i = 1; i < NumStates; ++i)
				m_states[i] = _initialState[i - 1];
		}

		auto operator()(const State& _state)
		{
			for (size_t i = 0; i < NumStates-1; ++i)
				m_states[i] = m_states[i + 1];
			m_states[NumStates - 1] = _state;

			InputMaker inputMaker;
			torch::Tensor input = inputMaker(m_system, m_states.data(), NumStates, 1, m_options);
			torch::Tensor next = m_network->forward(input);

			constexpr size_t stateSize = systems::sizeOfState<System>();
			const int64_t resultOffset = next.numel() / stateSize - 1;
			State state = *reinterpret_cast<State*>(next.data_ptr<T>() + resultOffset * stateSize);
			return state;
		}
	private:
		System m_system;
		Network& m_network;
		std::array<State, NumStates> m_states; // series of previous time steps
		c10::TensorOptions m_options;
	};

	// Helper to initialize the nn::Integrator
	template<size_t NumStates, typename State, typename Integrator>
	static std::pair< std::array<State, NumStates - 1>, State> computeTimeSeries(Integrator& _integrator, State _initialState)
	{
		std::array<State, NumStates - 1> initialStates;
		if constexpr (NumStates > 1)
		{
			initialStates[0] = _initialState;
			for (size_t i = 1; i < initialStates.size(); ++i)
			{
				initialStates[i] = _integrator(initialStates[i - 1]);
			}
			_initialState = _integrator(initialStates.back());
		}

		return { initialStates, _initialState };
	}
}
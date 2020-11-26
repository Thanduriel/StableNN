#pragma once

#include "../systems/state.hpp"
#include <torch/torch.h>
#include <c10/core/ScalarType.h>
#include <array>

namespace nn{

	template<typename System, typename Network, size_t NumStates = 1>
	class Integrator
	{
		using State = typename System::State;
		using T = typename System::ValueT;
	public:
		Integrator(Network& _network, const std::array<State, NumStates-1>& _initialState = {})
			: m_network(_network)
		{
			_network->to(c10::CppTypeToScalarType<T>());
			for (size_t i = 1; i < NumStates; ++i)
				m_states[i] = _initialState[i - 1];
		}

		template<typename State>
		auto operator()(const State& _state)
		{
			for (size_t i = 0; i < NumStates-1; ++i)
				m_states[i] = m_states[i + 1];
			m_states[NumStates - 1] = _state;

			constexpr size_t stateSize = sizeof(State) / sizeof(T);

			torch::Tensor next = m_network->forward(torch::from_blob(m_states.data(),
				{ static_cast<int64_t>(stateSize * NumStates) },
				c10::TensorOptions(c10::CppTypeToScalarType<T>())));

			const int64_t resultOffset = next.numel() / stateSize - 1;
			return *reinterpret_cast<State*>(next.data_ptr<T>() + resultOffset * stateSize);
		}
	private:
		std::array<State, NumStates> m_states; // series of previous time steps
		Network& m_network;
	};
}
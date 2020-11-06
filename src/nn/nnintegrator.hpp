#pragma once

#include "../systems/state.hpp"
#include <torch/torch.h>
#include <ATen/ScalarType.h>

namespace nn{

	template<typename Network, typename T>
	class Integrator
	{
	public:
		Integrator(Network& _network) : m_network(_network) 
		{
			_network.to(c10::CppTypeToScalarType<T>());
		}

		template<typename State>
		auto operator()(const State& _state) const
		{
			torch::Tensor next = m_network.forward(torch::from_blob((void*)(&_state), 
				{ static_cast<int64_t>(sizeof(State) / sizeof(T)) },
				c10::TensorOptions(c10::CppTypeToScalarType<T>()/*c10::ScalarType::Double*/)));
			return *reinterpret_cast<State*>(next.data<T>());
		}
	private:
		Network& m_network;
	};
}
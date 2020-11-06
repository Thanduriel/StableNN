#pragma once

#include <torch/torch.h>
#include <ATen/ScalarType.h>

namespace nn{

	template<typename Network, typename State>
	class Integrator
	{
	public:
		Integrator(Network& _network) : m_network(_network) 
		{
			_network.to(CppTypeToScalarType<typename State::ValueT>());
		}

		State operator()(const State& _state) const
		{
			torch::Tensor next = m_network.forward(torch::from_blob((void*)(&_state), { 2 },
				c10::TensorOptions(c10::ScalarType::Double)));
			return *reinterpret_cast<State*>(next.data<double>());
		}
	private:
		Network& m_network;
	};
}
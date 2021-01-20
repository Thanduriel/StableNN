#pragma once

#include <array>
#include <iostream>
//#include <torch/torch.h>

namespace systems {

	template<typename T, std::size_t Size>
	using State = std::array<T, Size>;

	// Assuming the state only consists of floating point numbers
	template<typename System>
	constexpr std::size_t sizeOfState()
	{
		static_assert(sizeof(typename System::State) % sizeof(typename System::ValueT) == 0,
			"Inconsistent state size.");
		return sizeof(typename System::State) / sizeof(typename System::ValueT);
	}

/*	template<typename System>
	torch::Tensor toTensor(const System::State& _state)
	{

	}*/
}

template<typename T, std::size_t Size>
std::ostream& operator << (std::ostream& out, const systems::State<T, Size>& s)
{
	for (const T& el : s)
		std::cout << el << " ";
	return out;
}
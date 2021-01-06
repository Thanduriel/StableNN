#pragma once

#include <array>
#include <iostream>

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
}

template<typename T, std::size_t Size>
std::ostream& operator << (std::ostream& out, const systems::State<T, Size>& s)
{
	for (const T& el : s)
		std::cout << el << " ";
	return out;
}
#pragma once

#include <array>
#include <iostream>

namespace systems {


	// using std::array as a simple mathematical vector
	template<typename T, std::size_t N>
	using Vec = std::array<T,N>;

	template<typename T, std::size_t N>
	Vec<T, N> operator+(const Vec<T, N>& a, const Vec<T, N>& b)
	{
		Vec<T, N> result;
		for (size_t i = 0; i < N; ++i)
			result[i] = a[i] + b[i];
		return result;
	}

	template<typename T, std::size_t N>
	Vec<T, N> operator*(const T s, const Vec<T, N>& v)
	{
		Vec<T, N> result;
		for (size_t i = 0; i < N; ++i)
			result[i] = s * v[i];
		return result;
	}

	template<typename T, std::size_t N>
	Vec<T, N> operator*(const Vec<T, N>& v, const T s)
	{
		return s * v;
	}

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
std::ostream& operator << (std::ostream& out, const systems::Vec<T, Size>& s)
{
	for (const T& el : s)
		std::cout << el << " ";
	return out;
}
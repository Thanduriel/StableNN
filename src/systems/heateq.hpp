#pragma once

#include <array>

namespace systems {

	// 1D heat equation with equidistant spatial discretization on a circle
	template<typename T, int N>
	class HeatEquation
	{
	public:
		using ValueT = T;
		using State = std::array<T, N>;
		constexpr static int NumPoints = N;

		HeatEquation(T _radius = 1.0) : m_radius(_radius) { m_heatCoefficients.fill(1.0); }
		HeatEquation(const std::array<T, N>& _heatCoefficients, T _radius = 1.0) 
			: m_heatCoefficients(_heatCoefficients), m_radius(_radius) {}

		T energy(const State& _state) const
		{
			T e = 0.0;
			for (T d : _state)
				e += d * d;

			return e;
		}

		// return actual index with respect to circular domain
		constexpr size_t index(size_t i) const
		{
			return (i + N) % N;
		}

		const T radius() const { return m_radius; }
		const std::array<T, N>& heatCoefficients() const { return m_heatCoefficients; }
	private:
		std::array<T, N> m_heatCoefficients;
		T m_radius;
	};
}
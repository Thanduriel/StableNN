#pragma once

#include <array>

namespace systems {

	// 1D heat equation with equidistant spatial discretization on a circle
	// with radius = 2 pi
	template<typename T, int N>
	class HeatEquation
	{
	public:
		using ValueT = T;
		using State = std::array<T, N>;
		constexpr static int NumPoints = N;

		HeatEquation() { m_heatCoefficients.fill(1.0); }
		HeatEquation(const std::array<T, N>& _heatCoefficients) : m_heatCoefficients(_heatCoefficients) {}

		T energy(const State& _state) const
		{
			T e = 0.0;
			for (T d : _state)
				e += d * d;

			return e;
		}

		const std::array<T, N>& getHeatCoefficients() const { return m_heatCoefficients; }
	private:
		std::array<T, N> m_heatCoefficients;
	};
}
#pragma once

namespace systems {

	// 1D heat equation with equidistant spatial discretization on a circle
	// with radius = 2 pi
	template<typename T, int N>
	class HeatEquation
	{
	public:
		using ValueT = T;
		using State = std::array<T, N>;

		HeatEquation() = default;

		T energy(const State& _state) const
		{
			T e = 0.0;
			for (T d : _state)
				e += d * d;

			return e;
		}
	private:
	};
}
#pragma once

namespace systems {

	// 1D heat equation with equidistant spatial discretization on a circle
	template<typename T, int N>
	class HeatEquation
	{
	public:
		HeatEquation() = default;

		using State = std::array<T, N>;

		T energy(const State& _state) const
		{
			T e = 0.0;
			for (double d : _state)
				e += d * d;

			return e;
		}
	private:
	};
}
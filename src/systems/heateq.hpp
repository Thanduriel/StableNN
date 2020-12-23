#pragma once

namespace systems {

	// 1D heat equation with equidistant spatial discretization on a circle
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
			for (double d : _state)
				e += d * d;

			return e;
		}
	private:
	};

	template<typename System>
	struct FiniteDifferences
	{
		using T = typename System::ValueT;
		using State = typename System::State;

		FiniteDifferences(T _dt) : m_dt(_dt) {}

		State operator()(const State& _state)
		{
			State next{};

			const T h = 1.0 / next.size();

			for (size_t i = 0; i < _state.size(); ++i)
			{
				const size_t pre = i > 0 ? i - 1 : _state.size() - 1;
				const size_t suc = i < _state.size() - 1 ? i + 1 : 0;
				next[i] = _state[i] + m_dt / (h * h) * (_state[pre] - 2.0 * _state[i] + _state[suc]);
			}

			return next;
		}

	private:
		T m_dt;
	};
}
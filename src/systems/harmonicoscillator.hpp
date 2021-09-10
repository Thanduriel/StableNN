#pragma once

#include "state.hpp"

#include <vector>
#include <iostream>
#include <cassert>

namespace systems {

	template<typename T>
	class HarmonicOscillator
	{
	public:
		using ValueT = T;
		using State = Vec<T, 2>;

		HarmonicOscillator(T _spring = 1.0, T _damping = 0.0)
			: m_spring(_spring), m_damping(_damping)
		{}

		Vec<T, 2> rhs(const State& _state) const
		{
			return { _state[1], -m_spring * _state[0] - m_damping * _state[1] };
		}

		T energy(const State& _state) const
		{
			return static_cast<T>(0.5) * m_spring * _state[0] * _state[0];
		}

		Vec<T, 2> toInitialState(T _amplitude, T _phaseShift)
		{
			const T cosTheta = std::cos(_phaseShift);
			return {_amplitude * cosTheta,
				_amplitude * (-m_damping/2 * cosTheta + frequency() * std::sin(_phaseShift))};
		}

		T spring() const { return m_spring; }
		T damping() const { return m_damping; }
		T frequency() const 
		{
			return std::sqrt(m_spring * m_spring - m_damping * m_damping / 4);
		}
	private:
		T m_spring;
		T m_damping;
	};

}
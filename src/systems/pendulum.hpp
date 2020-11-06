#pragma once

#include "state.hpp"
#include <cmath>

namespace systems {

	template<typename T>
	class Pendulum
	{
	public:
		using ValueT = T;

		struct State
		{
			T position;
			T velocity;
		};

		Pendulum(T _mass, T _gravity, T _len)
			: m_mass(_mass), m_gravity(_gravity), m_length(_len)
		{}

		T rhs (const State& _state) const
		{
			return -m_gravity / m_length * std::sin(_state.position);
		}

		T energy(const State& _state) const
		{
			return 0.5 * m_mass * m_length * m_length * _state.velocity * _state.velocity
				+ m_mass * m_gravity * m_length * (1.0 - std::cos(_state.position));
		}

	private:
		T m_mass;
		T m_gravity;
		T m_length;
	};

}
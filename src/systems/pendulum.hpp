#pragma once

#include "state.hpp"
#include "../constants.hpp"
#include <cmath>
#include <vector>
#include <iostream>
#include <cassert>

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

			constexpr size_t size() const { return 2; }
			constexpr T& operator[](size_t i) { assert(i < 2); return i == 0 ? position : velocity; }
			constexpr T operator[](size_t i) const { assert(i < 2); return i == 0 ? position : velocity; }

			friend std::ostream& operator << (std::ostream& out, const State& s)
			{
				out << std::fmod(s.position, PI) << ", " << s.velocity;
				return out;
			}
		};

		Pendulum(T _mass = 1.0, T _gravity = 1.0, T _len = 1.0)
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

		State energyToState(T _potentialEnergy, T _kineticEnergy) const
		{
			const T p = std::acos(1.0 - _potentialEnergy / (m_mass * m_gravity * m_length));
			const T v = std::sqrt(2.0 * _kineticEnergy / (m_mass * m_length * m_length));

			return { p,v };
		}

		T mass() const { return m_mass; }
		T gravity() const { return m_gravity; }
		T length() const { return m_length; }
	private:
		T m_mass;
		T m_gravity;
		T m_length;
	};

}
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

			constexpr State() : position(0.0), velocity(0.0) {}
			constexpr State(std::initializer_list<T> values) : position(*values.begin()), velocity(*(values.begin()+1)) {}
			constexpr State(const Vec<T,2>& values) : position(values[0]), velocity(values[1]) {}
			
			constexpr operator Vec<T, 2>() const
			{
				return { position, velocity };
			}

			State operator+(const Vec<T, 2>& step) const
			{
				return { position + step[0], velocity + step[1] };
			}

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

		Vec<T,2> rhs (const State& _state) const
		{
			return { _state.velocity, -m_gravity / m_length * std::sin(_state.position) };
		}

		T energy(const State& _state) const
		{
			return 0.5 * m_mass * m_length * m_length * _state.velocity * _state.velocity
				+ m_mass * m_gravity * m_length * (1.0 - std::cos(_state.position));
		}

		// construct a state with the specified amount of energy
		// the values will always be nonnegative
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
#pragma once

#include "state.hpp"

namespace systems {
namespace discretization{

	template<int Order>
	using RKMatrix = std::array<double, (Order - 1)* (Order - 1)>;
	template<int Order>
	using RKWeights = std::array<double, Order>;
	template<int Order>
	using RKNodes = std::array<double, Order - 1>;

	struct RK2_midpoint
	{
		constexpr static int Order = 2;

		constexpr static RKMatrix<Order> coefficients()
		{
			return { 0.5 };
		}

		constexpr static RKNodes<Order> nodes()
		{
			return { 0.5 };
		}

		constexpr static RKWeights<Order> weights()
		{
			return { 0.0, 1.0 };
		}
	};

	struct RK3
	{
		constexpr static int Order = 3;

		constexpr static RKMatrix<Order> coefficients()
		{
			return { 0.5, 0.5,
					-1.0, 2.0};
		}

		constexpr static RKNodes<Order> nodes()
		{
			return { 0.5, 1.0 };
		}

		constexpr static RKWeights<Order> weights()
		{
			return { 1/6.0, 2/3.0, 1/6.0 };
		}
	};

	struct RK4
	{
		constexpr static int Order = 4;

		constexpr static RKMatrix<Order> coefficients()
		{
			return { 0.5, 0.0, 0.0, 
					 0.0, 0.5, 0.0,
					 0.0, 0.0, 1.0};
		}

		constexpr static RKNodes<Order> nodes()
		{
			return { 0.5f, 0.5f, 1.0 };
		}

		constexpr static RKWeights<Order> weights()
		{
			return { 1/6.0, 1/3.0, 1/3.0, 1/6.0 };
		}
	};


	template<typename Params>
	class RungeKutta
	{
	public:
		constexpr static int Order = Params::Order;

		constexpr static RKMatrix<Order> coefficients = Params::coefficients();
		constexpr static RKNodes<Order> nodes = Params::nodes();
		constexpr static RKWeights<Order> weights = Params::weights();

		template<typename System, typename State, typename T>
		State operator()(const System& _system, const State& _state, T _dt) const
		{
			using Vec = decltype(_system.rhs(_state));

			std::array<Vec, Order> samples;

			for (int i = 0; i < Order; ++i)
			{
				const int offset = (i-1) * (Order - 1);
				Vec dir{};

				for (int j = 0; j < i; ++j)
					dir = dir + coefficients[offset + j] * samples[j];
				samples[i] = _system.rhs(_state + _dt * dir);
			}

			State sum = _state;
			for (int i = 0; i < Order; ++i)
				sum = sum +  _dt * weights[i] * samples[i];

			return sum;
		}
	};

}}
#pragma once

#include <utility>
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

	struct RK2_heun
	{
		constexpr static int Order = 2;

		constexpr static RKMatrix<Order> coefficients()
		{
			return { 1.0 };
		}

		constexpr static RKNodes<Order> nodes()
		{
			return { 1.0 };
		}

		constexpr static RKWeights<Order> weights()
		{
			return { 0.5, 0.5 };
		}
	};

	struct RK2_ralston
	{
		constexpr static int Order = 2;

		constexpr static RKMatrix<Order> coefficients()
		{
			return { 2.0/3.0 };
		}

		constexpr static RKNodes<Order> nodes()
		{
			return { 2.0/3.0 };
		}

		constexpr static RKWeights<Order> weights()
		{
			return { 0.25, 0.75 };
		}
	};

	struct RK3
	{
		constexpr static int Order = 3;

		constexpr static RKMatrix<Order> coefficients()
		{
			return { 0.5, 0.0,
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

	struct RK3_ralston
	{
		constexpr static int Order = 3;

		constexpr static RKMatrix<Order> coefficients()
		{
			return { 0.5, 0.0,
					 0.0, 0.75 };
		}

		constexpr static RKNodes<Order> nodes()
		{
			return { 0.5, 0.75 };
		}

		constexpr static RKWeights<Order> weights()
		{
			return { 2 / 9.0, 1 / 3.0, 4 / 9.0 };
		}
	};

	struct RK3_ssp
	{
		constexpr static int Order = 3;

		constexpr static RKMatrix<Order> coefficients()
		{
			return { 1.0, 0.0,
					 0.25, 0.25 };
		}

		constexpr static RKNodes<Order> nodes()
		{
			return { 1.0, 0.5 };
		}

		constexpr static RKWeights<Order> weights()
		{
			return { 1 / 6.0, 1 / 6.0, 2 / 3.0 };
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
			return { 0.5, 0.5, 1.0 };
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
			return integrate(_system, _state, _dt, std::make_index_sequence<Order>{});
	/*		using Vec = decltype(_system.rhs(_state));

			std::array<Vec, Order> samples;

			for (int i = 0; i < Order; ++i)
			{
				const int offset = (i-1) * (Order - 1);
				Vec dir{};

				for (int j = 0; j < i; ++j)
					dir = dir + coefficients[offset + j] * samples[j];
				samples[i] = _system.rhs(_state + _dt * dir);
			}

			Vec sum{};
			for (int i = 0; i < Order; ++i)
				sum = sum + weights[i] * samples[i];

			return _state + _dt * sum; */
		}

	private:
		template<typename System, typename State, typename T, std::size_t... I>
		State integrate(const System& _system, const State& _state, T _dt, std::index_sequence<I...>) const
		{
			using Vec = decltype(_system.rhs(_state));

			std::array<Vec, Order> samples;

			// comma operator ensures correct sequencing
			((samples[I] = sample<I>(_system, _state, _dt, samples)), ...);

			const Vec sum = ((weights[I] * samples[I]) + ...);

			return _state + _dt * sum;
		}

		template<size_t Offset, size_t I, size_t MaxInd, typename Vec>
		Vec computeSampleLocation(const std::array<Vec, Order>& _samples) const
		{
			constexpr size_t Ind = Offset + I;
			if constexpr (I < MaxInd - 1)
			{
				// skip 0 coefficients
				if constexpr(coefficients[Ind] != 0)
					return coefficients[Ind] * _samples[I] + computeSampleLocation<Offset, I + 1, MaxInd>(_samples);
				else
					return computeSampleLocation<Offset, I + 1, MaxInd>(_samples);
			}
			
			return coefficients[Ind] * _samples[I];
		}

		template<size_t I, typename System, typename State, typename Vec, typename T>
		Vec sample(const System& _system, const State& _state, T _dt, const std::array<Vec, Order>& _samples) const
		{
			if constexpr (I == 0)
				return _system.rhs(_state);
			else // this else is necessary to prevent code from being generated with Offset < 0
			{
				constexpr size_t Offset = (I - 1) * (Order - 1);
				const Vec dir = computeSampleLocation<Offset, 0, I>(_samples);

				return _system.rhs(_state + _dt * dir);
			}
		}
	};

}}
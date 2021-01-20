#pragma once

#include "heateq.hpp"
#include "../constants.hpp"
#include <torch/torch.h>
#include <c10/core/ScalarType.h>

namespace systems {
namespace discretization {
	// solver using the discrete fourier transform
	template<typename T, int N>
	struct AnalyticHeatEq
	{
		using State = typename HeatEquation<T,N>::State;

		// The default _initialState should only be used if operator() is never called.
		AnalyticHeatEq(const HeatEquation<T, N>& _system, T _dt, const State& _initialState = {})
			: m_dt(_dt)
		{
			const torch::Tensor s = torch::from_blob(const_cast<T*>(_initialState.data()),
				{ static_cast<int64_t>(_initialState.size()) },
				c10::TensorOptions(c10::CppTypeToScalarType<T>()));
			m_initialStateF = torch::fft_rfft(s, _initialState.size());
		}

		State operator()(const State& _state) 
		{
			m_t += m_dt;
			torch::Tensor scale = torch::zeros_like(m_initialStateF);
			for (int64_t i = 0; i < scale.size(0); ++i)
			{
				scale[i] = std::exp(-i*i * m_t);
			}

			const torch::Tensor nextF = scale * m_initialStateF;
			const torch::Tensor next = torch::fft_irfft(nextF, _state.size());

			return *reinterpret_cast<State*>(next.data_ptr<T>());
		}

		T getDeltaTime() const { return m_dt; }
	private:
		T m_dt;
		T m_t = 0.0;
		torch::Tensor m_initialStateF;
	};

	template<typename T, int N, int Dif = 1>
	struct FiniteDifferencesHeatEq
	{
		using State = typename HeatEquation<T, N>::State;

		FiniteDifferencesHeatEq(const HeatEquation<T, N>& _system, T _dt) 
			: m_system(_system), 
			m_dt(_dt), 
			m_r(0.0)
		{
			// double sized intervals so that the first order central differences uses whole steps
			const T h = 2.0 * PI / N;
			const T hSqr = (Dif * Dif * h * h);
			m_r = m_dt / hSqr;
			// stability criteria
			assert(m_r <= 0.5);
		}

		State operator()(const State& _state) const
		{
			State next;
			const auto& a = m_system.getHeatCoefficients();

			for (size_t i = 0; i < next.size(); ++i)
			{
				const T dxx = a[i] * (_state[index(i - Dif)] - 2.0 * _state[i] + _state[index(i + Dif)]);

				const size_t i_0 = index(i - 1);
				const size_t i_2 = index(i + 1);
				const T dx = (a[i_0] - a[i_2]) * (_state[i_0] - _state[i_2]);
				next[i] = _state[i] + m_r * (dxx + dx);
			}

			return next;
		}

	private:
		// return actual index with respect to circular domain
		size_t index(size_t i) const
		{
			return (i+N) % N;
		}

		const HeatEquation<T, N>& m_system;
		T m_r;
		T m_dt;
	};

	template<typename T, int N, int Dif = 1>
	struct FiniteDifferencesImplicit
	{
		using State = typename HeatEquation<T, N>::State;

		FiniteDifferencesImplicit(const HeatEquation<T, N>& _system, T _dt)
			: m_system(_system),
			m_dt(_dt),
			m_r(0.0)
		{
			// double sized intervals so that the first order central differences uses whole steps
			const T h = 2.0 * PI / N;
			const T hSqr = (Dif * Dif * h * h);
			m_r = m_dt / hSqr;

			const auto& a = m_system.getHeatCoefficients();

			constexpr int64_t m = N;
			auto options = c10::TensorOptions(c10::CppTypeToScalarType<T>());
			const T f = Dif == 1 ? 1.0 : -1.0;
			const torch::Tensor kernel = -m_r * torch::tensor({1.0, 1.0, -2.0, f, 1.0}, options);
			torch::Tensor mat = torch::zeros({ m, m }, options);
			for (int64_t i = 0; i < N; ++i)
			{
				const T da = a[index(i + 1)] - a[index(i - 1)];
				const torch::Tensor scale = Dif == 1 ? 
					torch::tensor({ 0.0, a[i], a[i], a[i], 0.0 })
					 : torch::tensor({ a[i], da, a[i], da, a[i] });
				
				for (int64_t j = 0; j < 5; ++j)
				{
					int64_t idx = static_cast<int64_t>(index(i + j - 2));
					mat.index_put_({ i,idx }, kernel[j] * scale[j]);
				}
			}
			m_luDecomposition = torch::_lu_with_info(torch::eye(m, options) + mat);
		}

		State operator()(const State& _state) const
		{
			auto options = c10::TensorOptions(c10::CppTypeToScalarType<T>());

			auto b = torch::from_blob(const_cast<T*>(_state.data()), { 1, static_cast<int64_t>(N), 1 }, options);
		//	std::cout << std::get<0>(m_luDecomposition);
		//	std::cout << std::get<1>(m_luDecomposition);
			torch::Tensor x = torch::lu_solve(b, std::get<0>(m_luDecomposition), std::get<1>(m_luDecomposition));

			return *reinterpret_cast<State*>(x.data_ptr<T>());
		}

	private:
		// return actual index with respect to circular domain
		size_t index(size_t i) const
		{
			return (i + N) % N;
		}

		const HeatEquation<T, N>& m_system;
		std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> m_luDecomposition;
		T m_r;
		T m_dt;
	};
}}
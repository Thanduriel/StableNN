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
			// double sized intervals so that the first order central differences uses whole steps;
			const T h = 2.0 * PI / N;
			m_r = m_dt / (Dif * Dif * h * h);
		}

		State operator()(const State& _state) const
		{
			State next;
			const auto& a = m_system.getHeatCoefficients();

			for (size_t i = 0; i < _state.size(); ++i)
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
}}
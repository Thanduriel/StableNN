#pragma once

#include "heateq.hpp"
#include <torch/torch.h>
#include <c10/core/ScalarType.h>

namespace systems {
namespace discretization {
	// solver using the discrete fourier transform
	template<typename T, int N>
	struct AnalyticHeatEq
	{
		using State = typename HeatEquation<T,N>::State;

		AnalyticHeatEq(const HeatEquation<T,N>& _system, T _dt) : m_dt(_dt) {}

		State operator()(const State& _state)
		{
			if (!m_initialized)
			{
				const torch::Tensor s = torch::from_blob(const_cast<T*>(_state.data()),
					{ static_cast<int64_t>(_state.size()) },
					c10::TensorOptions(c10::CppTypeToScalarType<T>()));
				m_initialStateF = torch::fft_rfft(s, _state.size());
				m_initialized = true;
			}

			m_t += m_dt;
			torch::Tensor scale = torch::zeros_like(m_initialStateF);
			for (int64_t i = 0; i < scale.size(0); ++i)
			{
				scale[i] = std::exp(-i * i * m_t);
			}

			const torch::Tensor nextF = scale * m_initialStateF;
			const torch::Tensor next = torch::fft_irfft(nextF, _state.size());

			return *reinterpret_cast<State*>(next.data_ptr<T>());
		}

	private:
		T m_dt;
		bool m_initialized = false;
		T m_t = 0.0;
		torch::Tensor m_initialStateF;
	};

	template<typename T, int N>
	struct FiniteDifferencesHeatEq
	{
		using State = typename HeatEquation<T, N>::State;

		FiniteDifferencesHeatEq(const HeatEquation<T, N>& _system, T _dt) : m_dt(_dt) {}

		State operator()(const State& _state)
		{
			State next{};

			const T h = 2.0 * PI / next.size();
			const T r = m_dt / (h * h);

			for (size_t i = 0; i < _state.size(); ++i)
			{
				const size_t pre = i > 0 ? i - 1 : _state.size() - 1;
				const size_t suc = i < _state.size() - 1 ? i + 1 : 0;
				next[i] = _state[i] + r * (_state[pre] - 2.0 * _state[i] + _state[suc]);
			}

			return next;
		}

	private:
		T m_dt;
	};
}}
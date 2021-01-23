#pragma once

#include "heateq.hpp"
#include "state.hpp"
#include "../constants.hpp"
#include <torch/torch.h>
#include <c10/core/ScalarType.h>

namespace systems {
namespace discretization {
	// solver using the discrete Fourier transform
	template<typename T, int N>
	struct AnalyticHeatEq
	{
		using State = typename HeatEquation<T,N>::State;

		// The default _initialState should only be used if operator() is never called.
		AnalyticHeatEq(const HeatEquation<T, N>& _system, T _dt, const State& _initialState = {})
			: m_system(_system),
			m_dt(_dt),
			m_rSqr(_system.radius() * _system.radius())
		{
			// variable heat coefficients are currently not supported
			assert(std::all_of(m_system.heatCoefficients().begin(), m_system.heatCoefficients().end(), [](T v) { return v == 1.0; }));

			const torch::Tensor s = torch::from_blob(const_cast<T*>(_initialState.data()),
				{ static_cast<int64_t>(_initialState.size()) },
				c10::TensorOptions(c10::CppTypeToScalarType<T>()));
			m_initialStateF = torch::fft_rfft(s, _initialState.size());
		}

		State operator()(const State& _state) 
		{
			m_t += m_dt;
			
			const torch::Tensor scale = getGreenFn(m_t);
			const torch::Tensor nextF = scale * m_initialStateF;
			const torch::Tensor next = torch::fft_irfft(nextF, _state.size());

			return *reinterpret_cast<State*>(next.data_ptr<T>());
		}

		T getDeltaTime() const { return m_dt; }

		torch::Tensor getGreenFn(T _time) const
		{
			torch::Tensor scale = torch::zeros_like(m_initialStateF);
			for (int64_t n = 0; n < scale.size(0); ++n)
			{
				scale[n] = std::exp(-n * n * _time / m_rSqr);
			}

			return scale;
		}
	private:
		const HeatEquation<T, N>& m_system;
		T m_dt;
		T m_t = 0.0;
		T m_rSqr;
		torch::Tensor m_initialStateF;
	};

	template<typename T, int N, int Dif = 1>
	struct FiniteDifferencesExplicit
	{
		using State = typename HeatEquation<T, N>::State;

		FiniteDifferencesExplicit(const HeatEquation<T, N>& _system, T _dt)
			: m_system(_system), 
			m_dt(_dt), 
			m_r(0.0)
		{
			// double sized intervals so that the first order central differences uses whole steps
			const T h = m_system.radius() * 2.0 * PI / N;
			const T hSqr = (Dif * Dif * h * h);
			m_r = m_dt / hSqr;
			// stability criteria
			assert(m_r <= 0.5);
		}

		State operator()(const State& _state) const
		{
			State next;
			const auto& a = m_system.heatCoefficients();

			for (size_t i = 0; i < next.size(); ++i)
			{
				const T dxx = (_state[m_system.index(i - Dif)] - 2.0 * _state[i] + _state[m_system.index(i + Dif)]);

				const size_t i_0 = m_system.index(i - 1);
				const size_t i_2 = m_system.index(i + 1);
				const T dx = _state[i_0] - _state[i_2];
				next[i] = _state[i] + m_r * ((a[i_0] - a[i_2]) * dx + a[i] * dxx);
			}

			return next;
		}

	private:
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
			m_options(c10::CppTypeToScalarType<T>())
		{
			// double sized intervals so that the first order central differences uses whole steps
			const T h = m_system.radius() * 2.0 * PI / N;
			const T hSqr = (Dif * Dif * h * h);
			const T r = _dt / hSqr;

			const auto& a = m_system.heatCoefficients();

			constexpr int64_t m = N;
			const T f = Dif == 1 ? 1.0 : -1.0;
			const torch::Tensor kernel = -r * torch::tensor({1.0, 1.0, -2.0, f, 1.0}, m_options);
			torch::Tensor mat = torch::zeros({ m, m }, m_options);
			for (int64_t i = 0; i < N; ++i)
			{
				const T da = a[m_system.index(i + 1)] - a[m_system.index(i - 1)];
				const torch::Tensor scale = Dif == 1 ? 
					torch::tensor({ 0.0, a[i], a[i], a[i], 0.0 })
					 : torch::tensor({ a[i], da, a[i], da, a[i] });
				
				for (int64_t j = 0; j < 5; ++j)
				{
					int64_t idx = static_cast<int64_t>(m_system.index(i + j - 2));
					mat.index_put_({ i,idx }, kernel[j] * scale[j]);
				}
			}
			m_LU = torch::_lu_with_info(torch::eye(m, m_options) + mat);
		}

		State operator()(const State& _state) const
		{
			auto options = c10::TensorOptions(c10::CppTypeToScalarType<T>());

			const torch::Tensor b = torch::from_blob(const_cast<T*>(_state.data()), { 1, static_cast<int64_t>(N), 1 }, m_options);
			const torch::Tensor x = torch::lu_solve(b, std::get<0>(m_LU), std::get<1>(m_LU));

			return *reinterpret_cast<State*>(x.data_ptr<T>());
		}

	private:
		const HeatEquation<T, N>& m_system;
		std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> m_LU;
		c10::TensorOptions m_options;
	};

	// input maker for nn::Integrator
	struct MakeInputHeatEq
	{
		template<typename T, int N>
		torch::Tensor operator()(const HeatEquation<T,N>& _system,
			const typename HeatEquation<T, N>::State* _states,
			int64_t _numStates,
			int64_t _batchSize,
			const c10::TensorOptions& _options) const
		{
			assert(_numStates % _batchSize == 0);
		//	static_assert(NumStates == 1, "Currently time series are not supported.");
			constexpr int64_t stateSize = systems::sizeOfState<HeatEquation<T, N>>();
			const int64_t statesPerBatch = _numStates / _batchSize;

			torch::Tensor stateInp = torch::from_blob(const_cast<typename HeatEquation<T, N>::State*>(_states),
				{ _batchSize, 1, stateSize * statesPerBatch },
				_options);
			torch::Tensor sysInp = torch::from_blob(const_cast<T*>(_system.heatCoefficients().data()),
				{ 1, 1, stateSize },
				_options);
			if (statesPerBatch > 1 || _batchSize > 1)
			{
				sysInp = sysInp.repeat({ _batchSize, 1, statesPerBatch });
			}

			return torch::cat({ stateInp, sysInp }, 1);
		}
	};

}}
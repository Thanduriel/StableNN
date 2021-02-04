#pragma once

#include "heateq.hpp"
#include "state.hpp"
#include "../constants.hpp"
#include "../nn/utils.hpp"

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
		//	assert(std::all_of(m_system.heatCoefficients().begin(), m_system.heatCoefficients().end(), [](T v) { return v == 1.0; }));

			const torch::Tensor s = torch::from_blob(const_cast<T*>(_initialState.data()),
				{ static_cast<int64_t>(_initialState.size()) },
				c10::TensorOptions(c10::CppTypeToScalarType<T>()));
			m_initialStateF = torch::fft_rfft(s, _initialState.size());

			m_t = 0.0;
		}

		AnalyticHeatEq(const AnalyticHeatEq& _oth, const HeatEquation<T, N>& _system, const State& _initialState)
			: AnalyticHeatEq(_system, _oth.deltaTime(), _initialState)
		{}

		State operator()(const State& _state) 
		{
			m_t += m_dt;
			
			const torch::Tensor scale = getGreenFnFt(m_t);
			const torch::Tensor nextF = scale * m_initialStateF;
			const torch::Tensor next = torch::fft_irfft(nextF, _state.size());

			return nn::tensorToArray<T,N>(next);
		}

		T deltaTime() const { return m_dt; }

		// Green's function in the frequency domain
		torch::Tensor getGreenFnFt(T _time) const
		{
			torch::Tensor scale = torch::zeros_like(m_initialStateF);
			for (int64_t n = 0; n < scale.size(0); ++n)
			{
				scale[n] = std::exp(-n * n * _time / m_rSqr);
			}

			return scale;
		}

		// Green's function in the spatial domain (convolution kernel)
		torch::Tensor getGreenFn(T _time, int _size = N) const
		{
			// restore convolution
			torch::Tensor green = torch::fft_irfft(getGreenFnFt(_time), _size);
			// rotate so that peak is in the middle
			green = green.roll({ _size / 2 }, {0});
			return green;
		}
	private:
		const HeatEquation<T, N>& m_system;
		T m_dt;
		T m_t = 0.0;
		T m_rSqr;
		torch::Tensor m_initialStateF;
	};

	template<typename T>
	constexpr auto LAPLACE_STENCILS = std::make_tuple(
		std::array<T, 3>{ 1.0, -2.0, 1.0 },
	//	std::array<T, 5>{ -1.0, 16.0, -30.0, 16.0, -1.0 });
		std::array<T, 5>{ -1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0 });

	template<typename T, int N, int Order = 1>
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
			const T hSqr = (h * h);
			m_r = m_dt / hSqr;
			// stability criteria
			assert(m_r <= 0.5);
		}

		State operator()(const State& _state) const
		{
			State next;
			const auto& a = m_system.heatCoefficients();

			const auto& stencil = std::get<Order-1>(LAPLACE_STENCILS<T>);
			const size_t shift = stencil.size() / 2;

			for (size_t i = 0; i < next.size(); ++i)
			{
				T dxx = 0.0;
				for(size_t j = 0; j < stencil.size(); ++j)
					dxx += stencil[j] * _state[m_system.index(i + j - shift)];
			/*	const size_t i_0 = m_system.index(i - 1);
				const size_t i_2 = m_system.index(i + 1);
				const T dx = _state[i_0] - _state[i_2];*/

				next[i] = _state[i] + m_r * (/*(a[i_0] - a[i_2]) * dx +*/ a[i] * dxx);
			}

			return next;
		}

		T deltaTime() const { return m_dt; }
	private:
		const HeatEquation<T, N>& m_system;
		T m_r;
		T m_dt;
	};

	template<typename T, int N, int Order = 1>
	struct FiniteDifferencesImplicit
	{
		using State = typename HeatEquation<T, N>::State;

		FiniteDifferencesImplicit(const HeatEquation<T, N>& _system, T _dt)
			: m_system(_system),
			m_options(c10::CppTypeToScalarType<T>())
		{
			// double sized intervals so that the first order central differences uses whole steps
			const T h = m_system.radius() * 2.0 * PI / N;
			const T hSqr = (h * h);
			const T r = _dt / hSqr;

			const auto& a = m_system.heatCoefficients();

			constexpr int64_t m = N;
			const auto& stencil = std::get<Order - 1>(LAPLACE_STENCILS<T>);
			const size_t shift = stencil.size() / 2;

			const torch::Tensor kernel = -r * nn::arrayToTensor(stencil, m_options);
			torch::Tensor mat = torch::zeros({ m, m }, m_options);
			for (int64_t i = 0; i < N; ++i)
			{
				const T da = a[m_system.index(i + 1)] - a[m_system.index(i - 1)];
				const torch::Tensor scale = Order == 1 ? 
					torch::tensor({ a[i], a[i], a[i] }, m_options)
					 : torch::tensor({ a[i], a[i], a[i], a[i], a[i] }, m_options);
				
				for (size_t j = 0; j < stencil.size(); ++j)
				{
					int64_t idx = static_cast<int64_t>(m_system.index(i + j - shift));
					mat.index_put_({ i,idx }, kernel[j] * scale[j]);
				}
			}
			m_LU = torch::_lu_with_info(torch::eye(m, m_options) + mat);
		}

		State operator()(const State& _state) const
		{
			const torch::Tensor b = torch::from_blob(const_cast<T*>(_state.data()), { 1, static_cast<int64_t>(N), 1 }, m_options);
			const torch::Tensor x = torch::lu_solve(b, std::get<0>(m_LU), std::get<1>(m_LU));

			return *reinterpret_cast<State*>(x.data_ptr<T>());
		}

	private:
		const HeatEquation<T, N>& m_system;
		std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> m_LU;
		c10::TensorOptions m_options;
	};

	// Wrapper which increases the internal spartial and temporal sampling to reduce the error.
	// @param M internal spatial resolution
	template<typename T, int N, int M>
	class SuperSampleIntegrator
	{
		using SmallState = typename HeatEquation<T, N>::State;
		using LargeState = typename HeatEquation<T, M>::State;
		using BaseIntegrator = FiniteDifferencesExplicit<T, M, 1>;
	public:
		SuperSampleIntegrator(const HeatEquation<T, N>& _system, T _dt, const SmallState& _state = {}, int _sampleRate = 1)
			: m_system(),
			m_state{},
			m_sampleRate(_sampleRate),
			m_deltaTime(_dt),
			m_integrator(m_system, _dt / _sampleRate),
			m_options(c10::CppTypeToScalarType<T>())
		{
			// upscale heat coefficients via linear interpolation
			const auto& smallCoefs = _system.heatCoefficients();
			std::array<T, M> coefficients;
			const T ratio = static_cast<T>(N) / static_cast<T>(M);
			for (int i = 0; i < M; ++i)
			{
				const T current = i * ratio;
				const int lower = std::floor(current);
				const int upper = std::ceil(current);
				const T t = current - lower;

				coefficients[i] = smallCoefs[lower] * (1.0 - t) + smallCoefs[std::min(upper, N-1)] * t;
			}
			m_system = HeatEquation<T, M>(coefficients, _system.radius());
		/*	torch::Tensor small = nn::arrayToTensor(_system.heatCoefficients(), m_options);
			small = torch::fft_rfft(small);
			torch::Tensor large = torch::fft_irfft(small, M);
			m_system = HeatEquation<T,M>(nn::tensorToArray<T, M>(large), _system.radius());*/

			// upsale state with fft
			torch::Tensor stateSmall = nn::arrayToTensor(_state, m_options);
			stateSmall = torch::fft_rfft(stateSmall);
			torch::Tensor stateLarge = torch::fft_irfft(stateSmall, M);
			m_state = nn::tensorToArray<T, M>(stateLarge);
		}

		SuperSampleIntegrator(const SuperSampleIntegrator& _oth, const HeatEquation<T, N>& _system, const SmallState& _state)
			: SuperSampleIntegrator(_system, _oth.m_deltaTime, _state, _oth.m_sampleRate)
		{
		}

		SmallState operator()(const SmallState&)
		{
			for (int i = 0; i < m_sampleRate; ++i)
				m_state = m_integrator(m_state);

			return downscaleState(m_state);
		}

		SmallState downscaleState(const LargeState& _large) const
		{
			torch::Tensor stateLarge = torch::fft_rfft(nn::arrayToTensor(_large, m_options));
			torch::Tensor stateSmall = torch::fft_irfft(stateLarge, N);

			return nn::tensorToArray<T, N>(stateSmall);
		}

		T deltaTime() const { return m_deltaTime; }
		const HeatEquation<T, M>& internalSystem() const { return m_system; }
		const LargeState& internalState() const { return m_state; }
	private:
		HeatEquation<T, M> m_system;
		LargeState m_state;
		int m_sampleRate;
		T m_deltaTime;
		BaseIntegrator m_integrator;
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

			// extend system infos to same size as states
			if (statesPerBatch > 1 || _batchSize > 1)
			{
				sysInp = sysInp.repeat({ _batchSize, 1, statesPerBatch });
			}
			
			// combine system and state channel
			return torch::cat({ stateInp, sysInp }, 1);
		}
	};

}}
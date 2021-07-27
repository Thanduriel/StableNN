#pragma once

#include <functional>

namespace eval {

	class PendulumRenderer
	{
	public:
		using Integrator = std::function<double()>;
		PendulumRenderer(double _deltaTime, Integrator _integrator);
		void run();
	private:
		double m_deltaTime;
		std::function < double() > m_integrator;
	};

	class HeatRenderer
	{
	public:
		using Integrator = std::function < std::vector<double>() >;
		HeatRenderer(double _deltaTime, size_t _domainSize, 
			const double* _diffusivity, Integrator _integrator);

		void run();
	private:
		double m_deltaTime;
		size_t m_domainSize;
		std::vector<float> m_diffusivity;
		Integrator m_integrator;
	};
}
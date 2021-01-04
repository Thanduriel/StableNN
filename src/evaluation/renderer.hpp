#pragma once

#include <functional>

namespace eval {

	class PendulumRenderer
	{
	public:
		PendulumRenderer(double _deltaTime);

		void addIntegrator(std::function<double()> _integrator);
		void run();
	private:
		double m_deltaTime;
		std::function < double() > m_integrator;
	};

	class HeatRenderer
	{
	public:
		using Integrator = std::function < std::vector<double>() >;
		HeatRenderer(double _deltaTime, Integrator _integrator);

		void run();
	private:
		double m_deltaTime;
		Integrator m_integrator;
	};
}
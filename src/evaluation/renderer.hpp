#pragma once

#include <SFML/Graphics.hpp>
#include <functional>

namespace eval {

	class PendulumRenderer
	{
	public:
		PendulumRenderer(double _deltaTime);

		void addIntegrator(std::function<double()> _integrator);
		void run();
	private:
		sf::RenderWindow m_window;
		double m_deltaTime;

		std::function < double() > m_integrator;
	};
}
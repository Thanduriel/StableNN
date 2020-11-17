#include "renderer.hpp"
#include <SFML/Graphics.hpp>

namespace eval {

	PendulumRenderer::PendulumRenderer(double _deltaTime)
		: m_window(sf::VideoMode(512,512), "pendulum"),
		m_deltaTime(_deltaTime)
	{
		m_window.setFramerateLimit(static_cast<unsigned>(1.0 / _deltaTime));
	}

	void PendulumRenderer::addIntegrator(std::function<double()> _integrator)
	{
		m_integrator = _integrator;
	}

	void PendulumRenderer::run()
	{
		const sf::Vector2f origin(256.f, 256.f);
		const float length = 100.f;

		sf::CircleShape mass(16.f);
		mass.setOrigin({ 16.f,16.f });

		sf::RectangleShape line(sf::Vector2f(2.f, length));
		line.setOrigin({ 1.f, 0.f });
		line.setPosition(origin);

		while (m_window.isOpen())
		{
			sf::Event event;
			while (m_window.pollEvent(event))
			{
				if (event.type == sf::Event::Closed)
					m_window.close();
			}

			// update positions
			const float angle = m_integrator();
			mass.setPosition(origin + length * sf::Vector2f(std::sin(angle), std::cos(angle)));
			line.setRotation(-angle / 3.1415f * 180.f);

			m_window.clear(sf::Color::Black);

			m_window.draw(mass);
			m_window.draw(line);

			m_window.display();
		}
	}
}
#include "renderer.hpp"
#include <SFML/Graphics.hpp>
#include <cmath>

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
			sf::sleep(sf::seconds(m_deltaTime));
		}
	}

	HeatRenderer::HeatRenderer(double _deltaTime, Integrator _integrator)
		: m_window(sf::VideoMode(512, 512), "heateq"),
		m_deltaTime(_deltaTime),
		m_integrator(_integrator)
	{

	}

	void HeatRenderer::run()
	{
		const sf::Vector2f origin(256.f, 256.f);
		std::vector<double> state = m_integrator();

		// + 1 for origin, + 1 to close the loop
		sf::VertexArray triangles(sf::TriangleFan, state.size() + 2);
		triangles[0].position = origin;

		while (m_window.isOpen())
		{
			sf::Event event;
			while (m_window.pollEvent(event))
			{
				if (event.type == sf::Event::Closed)
					m_window.close();
			}

			for (size_t i = 0; i < state.size(); ++i)
			{
				const float angle = static_cast<float>(i) / state.size() * 2.f * 3.14159f;
				triangles[i + 1].position = origin 
					+ 10.f * static_cast<float>(state[i]) * sf::Vector2f(std::sin(angle), std::cos(angle));
			}
			triangles[state.size() + 1].position = triangles[1].position;

			m_window.clear(sf::Color::Black);
			m_window.draw(triangles);
			m_window.display();

			state = m_integrator();
			sf::sleep(sf::seconds(m_deltaTime));
		}
	}
}
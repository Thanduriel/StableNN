#include "renderer.hpp"
#include "../constants.hpp"
#include <SFML/Graphics.hpp>
#include <cmath>

namespace eval {

	PendulumRenderer::PendulumRenderer(double _deltaTime)
		: m_deltaTime(_deltaTime)
	{
	}

	void PendulumRenderer::addIntegrator(std::function<double()> _integrator)
	{
		m_integrator = _integrator;
	}

	void PendulumRenderer::run()
	{
		const sf::Vector2f origin(256.f, 256.f);
		const float length = 100.f;

		sf::RenderWindow window(sf::VideoMode(512, 512), "pendulum");
		window.setFramerateLimit(static_cast<unsigned>(1.0 / m_deltaTime));

		sf::CircleShape mass(16.f);
		mass.setOrigin({ 16.f,16.f });

		sf::RectangleShape line(sf::Vector2f(2.f, length));
		line.setOrigin({ 1.f, 0.f });
		line.setPosition(origin);

		while (window.isOpen())
		{
			sf::Event event;
			while (window.pollEvent(event))
			{
				if (event.type == sf::Event::Closed)
					window.close();
			}

			// update positions
			const float angle = m_integrator();
			mass.setPosition(origin + length * sf::Vector2f(std::sin(angle), std::cos(angle)));
			line.setRotation(-angle / PI_F * 180.f);

			window.clear(sf::Color::Black);

			window.draw(mass);
			window.draw(line);

			window.display();
		}
	}

	HeatRenderer::HeatRenderer(double _deltaTime, size_t _domainSize, const double* _diffusivity, Integrator _integrator)
		: m_deltaTime(_deltaTime),
		m_domainSize(_domainSize),
		m_diffusivity(_diffusivity, _diffusivity + _domainSize),
		m_integrator(_integrator)
	{
	//	m_diffusivity.resize(m_domainSize);
	//	std::copy(_diffusivity, _diffusivity + _domainSize, m_diffusivity.data());
	}

	constexpr float BASE_RADIUS = 8.f;

	void HeatRenderer::run()
	{
		const sf::Vector2f origin(256.f, 256.f);
		std::vector<double> state = m_integrator();

		sf::RenderWindow window(sf::VideoMode(512, 512), "heateq");
		window.setFramerateLimit(static_cast<unsigned>(1.0 / m_deltaTime));

		// + 1 for origin, + 1 to close the loop
		sf::VertexArray triangles(sf::TriangleFan, state.size() + 2);
		triangles[0].position = origin;
		triangles[0].color = sf::Color(255, 255, 255);

		int steps = 0;
		while (window.isOpen())
		{
			sf::Event event;
			while (window.pollEvent(event))
			{
				if (event.type == sf::Event::Closed)
					window.close();
			}

			for (size_t i = 0; i < state.size(); ++i)
			{
				const float angle = static_cast<float>(i) / state.size() * 2.f * PI_F;
				const sf::Vector2f dir = sf::Vector2f(std::sin(angle), std::cos(angle));
				triangles[i + 1].position = origin
					+ BASE_RADIUS * dir
					+ static_cast<float>(state[i]) * dir;

				const sf::Uint8 red = m_diffusivity[i] < 1.0 ? static_cast<sf::Uint8>(m_diffusivity[i] * 255) : 0;
				const sf::Uint8 blue = m_diffusivity[i] > 1.0 ? static_cast<sf::Uint8>((m_diffusivity[i]-1.0) * 255) : 0;
				triangles[i + 1].color = sf::Color(red, 255 - red, 255 - red);
			}
			triangles[state.size() + 1].position = triangles[1].position;
			triangles[state.size() + 1].color = triangles[1].color;

			window.clear(sf::Color::Black);
			window.draw(triangles);
			window.setTitle("heateq - " + std::to_string(steps) + " / " + std::to_string(steps * m_deltaTime) + "s");
			window.display();

			state = m_integrator();
			++steps;
		}
	}
}
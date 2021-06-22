#include "renderer.hpp"
#include "../constants.hpp"
#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Color.hpp>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace eval {

	struct HSV 
	{
		float h;       // angle in degrees
		float s;       // a fraction between 0 and 1
		float v;       // a fraction between 0 and 1
	};

	sf::Color HSVtoRGB(HSV in)
	{
		sf::Color out;
		if (in.s <= 0.0) // < is bogus, just shuts up warnings
		{
			out.r = in.v * 255;
			out.g = in.v * 255;
			out.b = in.v * 255;
			return out;
		}
		double hh = in.h;
		if (hh >= 360.0) hh = 0.0;
		hh /= 60.0;
		int i = hh;
		double ff = hh - i;

		const sf::Uint8 v = in.v * 255;
		const sf::Uint8 p = in.v * (1.0 - in.s) * 255;
		const sf::Uint8 q = in.v * (1.0 - (in.s * ff)) * 255;
		const sf::Uint8 t = in.v * (1.0 - (in.s * (1.0 - ff))) * 255;

		switch (i) {
		case 0:
			out.r = v;
			out.g = t;
			out.b = p;
			break;
		case 1:
			out.r = q;
			out.g = v;
			out.b = p;
			break;
		case 2:
			out.r = p;
			out.g = v;
			out.b = t;
			break;

		case 3:
			out.r = p;
			out.g = q;
			out.b = v;
			break;
		case 4:
			out.r = t;
			out.g = p;
			out.b = v;
			break;
		case 5:
		default:
			out.r = v;
			out.g = p;
			out.b = q;
			break;
		}

		return out;
	}

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
		m_diffusivity(_domainSize, 0.0),
		m_integrator(_integrator)
	{
		const auto [minDif, maxDif] = std::minmax_element(_diffusivity, _diffusivity + _domainSize);
		const double interval = (*maxDif - *minDif) * 0.5;

		if (interval > 1.e-6)
		{
			for (size_t i = 0; i < _domainSize; ++i)
				m_diffusivity[i] = static_cast<float>((_diffusivity[i] - *minDif) / interval - 1.0);
		}
	}

	constexpr int WINDOW_SIZE = 512;
	const float HALF_SIZE = WINDOW_SIZE * 0.5f;
	const float MEAN_RADIUS = WINDOW_SIZE * 0.25f;
	const float MIN_RADIUS = 8.f;

	void HeatRenderer::run()
	{
		std::vector<double> state = m_integrator();
		// scale system to fit the window
		const float mean = std::accumulate(state.begin(), state.end(), 0.0) / state.size();
		const auto [minE, maxE] = std::minmax_element(state.begin(), state.end());
		const float radiusScale = (HALF_SIZE - MIN_RADIUS) / (*maxE - *minE);
		const float baseRadius = std::max(MIN_RADIUS - static_cast<float>(*minE) * radiusScale, MEAN_RADIUS - mean);

		sf::RenderWindow window(sf::VideoMode(WINDOW_SIZE, WINDOW_SIZE), "heateq");
		const sf::Vector2f origin(HALF_SIZE, HALF_SIZE);
		window.setFramerateLimit(std::min(static_cast<unsigned>(1.0 / m_deltaTime), 60u));

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
					+ (baseRadius + radiusScale * static_cast<float>(state[i])) *dir;

				HSV color{ m_diffusivity[i] > 0.0 ? 0.f : 240.f, std::abs(m_diffusivity[i]), 1.f };
				triangles[i + 1].color = HSVtoRGB(color);
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
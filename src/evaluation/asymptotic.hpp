#pragma once

namespace eval {

	template<typename System, typename Integrator>
	void findAttractors(const System& _system, const Integrator& _integrator)
	{
		using State = typename System::State;
		System system(_system);
		Integrator integrator(_integrator);

		std::vector<double> attractors; // energy
		std::vector<std::pair<double, double>> repellers; // position [lower, upper]

		auto integrate = [&](double x)
		{
			State state{ x, 0.0 };
			for (int i = 0; i < 20000; ++i)
				state = integrator(state);

			return state;
		};

		const double stepSize = 3.14159 / 16.0;
		constexpr double THRESHOLD = 0.05;
		for (double x = 0.0; x < 3.14159; x += stepSize)
		{
			State state = integrate(x);
			const double e = std::min(2.0, system.energy(state));

			std::cout << "attractors: energy, position\n";
			if (attractors.empty() || std::abs(attractors.back() - e) > THRESHOLD)
			{
				std::cout << e << ", " << x << std::endl;
				attractors.push_back(e);
				repellers.push_back({ 0.0, x });
			}
			/*	double e0 = 0.0;
				double e1 = system.energy(state);
				do {
					e1 = e0;
					state = integrator(state);
					double e0 = system.energy(state);
				} while (e0 - e1 > 0.1);*/
		}

		// refine repellers / bifurcations
		for (size_t j = 1; j < repellers.size(); ++j)
		{
			double x = repellers[j].second;
			double step = stepSize * 0.5;
			double sign = -1.0;
			for (int i = 0; i < 8; ++i)
			{
				x += step * sign;
				State state = integrate(x);
				step *= 0.5;
				const double e = std::min(2.0, system.energy(state));
				if (std::abs(e - attractors[j]) > THRESHOLD)
				{
					sign = 1.0;
					repellers[j].first = x;
				}
				else
				{
					sign = -1.0;
					repellers[j].second = x;
				}
			}
		}
		std::cout << "repellers: \n";
		for (auto& [min, max] : repellers)
		{
			std::cout << "[" << min << ", " << max << "]" << "\n";
		}
	}

	// determine frequency of a periodic motion
	// @param _initialState should be the start of a period
	// @param _periods Number of periods to average over
	// @return frequency in periods / steps
	template<typename State, typename Integrator>
	double computePeriodLength(const State& _initialState, const Integrator& _integrator, int _periods = 1)
	{
		Integrator integrator(_integrator);
		State state(_initialState);

		size_t steps = 0;
		for (int period = 0; period < _periods; ++period)
		{
			State prev = state;
			while(true)
			{
				State next = integrator(state);
				if (state.position > next.position && state.position > prev.position)
					break;
				prev = state;
				state = next;
				++steps;
			}
		}

		return static_cast<double>(steps) / static_cast<double>(_periods);
	}
}
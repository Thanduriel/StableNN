#pragma once

namespace eval {

	// energy level indicating that the simulation grows unbounded
	constexpr double INF_ENERGY = 4.0;
	constexpr double THRESHOLD = 0.02;

	// @return energy of attractors and intervals bounding the position 
	//         of possible initial states for bifurcation points
	template<typename System, typename Integrator>
	auto findAttractors(const System& _system, const Integrator& _integrator, bool computeBifurcations = true)
		-> std::pair<std::vector<double>, std::vector<std::pair<double,double>>>
	{
		using State = typename System::State;
		System system(_system);
		Integrator integrator(_integrator);

		std::vector<double> attractors; // energy
		std::vector<std::pair<double, double>> repellers; // position [lower, upper]

		constexpr double stepSize = PI / 16.0;

		// integrate until asymptotic energy can be identified
		auto integrate = [&system, &integrator](double x)
		{
			State state{ x, 0.0 };
			double e0 = std::min(INF_ENERGY, system.energy(state));
			double d0 = 0.0;
			double d1 = 0.0;
			int steps = 0;
			do {
				d0 = d1;
				for (int i = 0; i < 5000; ++i)
					state = integrator(state);
				const double e1 = std::min(INF_ENERGY, system.energy(state));
				d1 = e1 - e0;
				e0 = e1;
				++steps;
				// switch of sign indicates oscillation around attractor
				// second rule is for infinity and 0
			} while (d0 * d1 >= 0.0 && d1 != 0.0 && steps < 256);//256

			return e0;
		};

		std::cout << "attractors: energy, position\n";
		for (double x = 0.0; x < PI; x += stepSize)
		{
			const double e = integrate(x);
			if (attractors.empty() || std::abs(attractors.back() - e) > THRESHOLD)
			{
				std::cout << e << ", " << x << std::endl;
				attractors.push_back(e);
				repellers.push_back({ 0.0, x });
			}
			// early out if simulation already explodes
			if (e == INF_ENERGY) break;
		}

		// refine repellers / bifurcations
		if (computeBifurcations)
		{
			for (size_t j = 1; j < repellers.size(); ++j)
			{
				double x = repellers[j].second;
				double step = stepSize * 0.5;
				double sign = -1.0;
				for (int i = 0; i < 8; ++i)
				{
					x += step * sign;
					step *= 0.5;

					constexpr int maxRetries = 4;
					double e0 = 0.0;
					int k = 0;
					for (; k < maxRetries; ++i)
					{
						const double e = integrate(x);
						if (std::abs(e - attractors[j - 1]) <= THRESHOLD)
						{
							sign = 1.0;
							repellers[j].first = x;
							break;
						}
						else if (std::abs(e - attractors[j]) <= THRESHOLD)
						{
							sign = -1.0;
							repellers[j].second = x;
							break;
						}
						if (e == e0) k = maxRetries;
						e0 = e;
					}
					if(k == maxRetries) // very close to the bifurcation
					{
						repellers[j].first = x;
						repellers[j].second = x;
						break;
					}
				}
			}
			for (auto& [min, max] : repellers)
			{
				std::cout << "[" << min << ", " << max << "]" << "\n";
			}
		}

		return { attractors, repellers };
	}

	// determine frequency of a periodic motion
	// @param _periods Number of periods to average over
	// @return period length in steps / periods or 0 if no period could be detected.
	template<typename State, typename Integrator>
	double computePeriodLength(const State& _initialState, const Integrator& _integrator, int _periods = 1, int _maxSteps = 20048)
	{
		Integrator integrator(_integrator);
		State state(_initialState);
		State prev = state;
		state = integrator(state);
		for(int i = 0; i < 512; ++i)
			state = integrator(state);

		size_t steps = 0;
		// one extra period because the first one may not be complete
		for (int period = 0; period <= _periods; ++period)
		{
			bool foundMax = false;
			for(int i = 0; i < _maxSteps; ++i)
			{
				State next = integrator(state);
				// local maximum
				if (state.position > next.position && state.position > prev.position)
				{
					foundMax = true;
					break;
				}
				prev = state;
				state = next;
				++steps;
			}
			if (!foundMax) return 0.0;
			prev = state;
			// reset steps at first peak
			if (!period) steps = 0;
		}

		return static_cast<double>(steps) / static_cast<double>(_periods);
	}
}
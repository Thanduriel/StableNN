#include "systems/heateq.hpp"
#include "systems/heateqsolver.hpp"
#include "evaluation/renderer.hpp"

using System = systems::HeatEquation<double, 64>;
using State = typename System::State;

int main()
{
	System heatEq;
//	systems::discretization::FiniteDifferencesHeatEq integ(heatEq, 0.0001);
	systems::discretization::AnalyticHeatEq integ(heatEq, 0.0001);
	System::State testState{};
	testState.fill(50.f);
	testState[32] = 272.0;
	testState[4] = 0.0;
	testState[5] = 0.0;
	testState[6] = 0.0;
	testState[42] = 78.0;

	int counter = 0;
	eval::HeatRenderer renderer(0.01, [&, state=testState]() mutable
		{
			state = integ(state);
			std::vector<double> exState;
			for (double d : state)
				exState.push_back(d);
			return exState;
		});
	renderer.run();
	return 0;
}
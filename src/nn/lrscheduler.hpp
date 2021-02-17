#pragma once

#include <torch/torch.h>

namespace nn {

	// Map optimizers to their respective options
	template<typename Optimizer>
	struct OptimzerToOptions;

	template<>
	struct OptimzerToOptions<torch::optim::Adam> { using value_t = torch::optim::AdamOptions; };

	template<>
	struct OptimzerToOptions<torch::optim::LBFGS> { using value_t = torch::optim::LBFGSOptions; };

	template<>
	struct OptimzerToOptions<torch::optim::SGD> { using value_t = torch::optim::SGDOptions; };

	template<>
	struct OptimzerToOptions<torch::optim::RMSprop> { using value_t = torch::optim::RMSpropOptions; };


	// Exponentially decaying learning rate.
	template<typename Optimizer>
	class LearningRateScheduler
	{
		using OptimizerOptions = typename OptimzerToOptions<Optimizer>::value_t;
	public:
		LearningRateScheduler(Optimizer& _optimizer, double _rate, int64_t _epochUpdate = 1)
			: m_optimizer(_optimizer), m_rate(_rate), m_epochUpdate(_epochUpdate)
		{}

		void step(int64_t _epoch) const
		{
			if (_epoch % m_epochUpdate != 0) return;

			for (auto& group : m_optimizer.param_groups())
			{
				if (group.has_options())
				{
					auto& options = static_cast<OptimizerOptions&>(group.options());
					options.lr(options.lr() * m_rate);
				}
			}
		}

	private:
		Optimizer& m_optimizer;
		double m_rate;
		int64_t m_epochUpdate;
	};
}
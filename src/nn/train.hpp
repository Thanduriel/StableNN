#pragma once

#include "nnmaker.hpp"
#include "utils.hpp"
#include "../defs.hpp"
#include "../generator.hpp"
#include "hyperparam.hpp"
#include <torch/torch.h>
#include <mutex>

namespace nn {

	template<typename Network, typename System, typename Integrator>
	struct TrainNetwork
	{
		using State = typename System::State;

		TrainNetwork(const System& _system, std::vector<State> _trainStates, std::vector<State> _validStates)
			: m_system(_system),
			m_trainStates(std::move(_trainStates)),
			m_validStates(std::move(_validStates))
		{
		}

		double operator()(const nn::HyperParams& _params) const
		{
			Integrator referenceIntegrator(m_system, *_params.get<double>("time_step") / HYPER_SAMPLE_RATE);
			DataGenerator<System, Integrator> generator(m_system, referenceIntegrator);

			namespace dat = torch::data;
			const size_t numInputs = _params.get<size_t>("num_inputs", NUM_INPUTS);
			auto dataset = generator.generate(m_trainStates, 16, HYPER_SAMPLE_RATE, numInputs, USE_SINGLE_OUTPUT, NUM_FORWARDS)
				.map(dat::transforms::Stack<>());
			auto validationSet = generator.generate(m_validStates, 16, HYPER_SAMPLE_RATE, numInputs, USE_SINGLE_OUTPUT, NUM_FORWARDS)
				.map(dat::transforms::Stack<>());

			// LBFGS does not work with mini batches
			using Sampler = std::conditional_t<USE_SEQ_SAMPLER,
				dat::samplers::SequentialSampler,
				dat::samplers::RandomSampler>;
			auto data_loader = dat::make_data_loader<Sampler>(
				dataset,
				dat::DataLoaderOptions().batch_size(USE_LBFGS ? std::numeric_limits< size_t>::max() : 64));
			auto validationLoader = dat::make_data_loader(
				validationSet,
				dat::DataLoaderOptions().batch_size(64));

			if constexpr (THREAD_FIXED_SEED)
			{
				m_initMutex.lock();
				torch::manual_seed(TORCH_SEED);
			}
			auto net = nn::makeNetwork<Network, USE_WRAPPER, 2>(_params);
			auto bestNet = nn::makeNetwork<Network, USE_WRAPPER, 2>(_params);
			if constexpr (THREAD_FIXED_SEED)
			{
				m_initMutex.unlock();
			}

			auto lossFn = [](const torch::Tensor& self, const torch::Tensor& target)
			{
				return nn::lp_loss(self, target, 3);
			};

			auto nextInput = [](const torch::Tensor& input, const torch::Tensor& output)
			{
				if constexpr (USE_SINGLE_OUTPUT && NUM_FORWARDS > 1)
					return nn::shiftTimeSeries(input, output, 2);
				else
					return output;
			};


			auto makeOptimizer = [&_params, &net]()
			{
				if constexpr (USE_LBFGS)
					return torch::optim::LBFGS(net->parameters(),
						torch::optim::LBFGSOptions(*_params.get<double>("lr")));
				else
					return torch::optim::Adam(net->parameters(),
						torch::optim::AdamOptions(_params.get<double>("lr", 3.e-4))
						.weight_decay(_params.get<double>("weight_decay", 1.e-6))
						.amsgrad(_params.get<bool>("amsgrad", false)));
			};
			auto optimizer = makeOptimizer();

			double bestValidLoss = std::numeric_limits<double>::max();

			//std::ofstream lossFile("loss.txt");

			for (int64_t epoch = 1; epoch <= 2048; ++epoch)
			{
				// train
				net->train();

				torch::Tensor totalLoss = torch::zeros({ 1 });

				for (torch::data::Example<>& batch : *data_loader)
				{
					auto closure = [&]()
					{
						net->zero_grad();
						torch::Tensor output;
						torch::Tensor input = batch.data;
						for (int64_t i = 0; i < NUM_FORWARDS; ++i)
						{
							output = net->forward(input);
							input = nextInput(input, output);
						}
						torch::Tensor loss = lossFn(output, batch.target);
						totalLoss += loss;

						loss.backward();
						return loss;
					};

					optimizer.step(closure);
				}

				// validation
				net->eval();
				torch::Tensor validLoss = torch::zeros({ 1 });
				for (torch::data::Example<>& batch : *validationLoader)
				{
					torch::Tensor output;
					torch::Tensor input = batch.data;
					for (int64_t i = 0; i < NUM_FORWARDS; ++i)
					{
						output = net->forward(input);
						input = nextInput(input, output);
					}
					torch::Tensor loss = lossFn(output, batch.target);
					validLoss += loss;
				}


				const double totalValidLossD = validLoss.item<double>();
				if (totalValidLossD < bestValidLoss)
				{
					if constexpr (MODE != Mode::TRAIN_MULTI)
						std::cout << validLoss.item<double>() << "\n";
					bestNet = nn::clone(net);
					bestValidLoss = totalValidLossD;
				}

				//	lossFile << totalLoss.item<double>() << ", " << totalLossD << "\n";
			//	std::cout << "finished epoch with loss: " << totalLoss.item<double>() << "\n";
			}
			if (LOG_LOSS)
			{
				std::unique_lock<std::mutex> lock(m_loggingMutex);
				std::ofstream lossLog("losses.txt", std::ios::app);
				lossLog << bestValidLoss << std::endl;
			}
			if (SAVE_NET)
			{
				torch::save(bestNet, _params.get<std::string>("name", "net.pt"));
			}

			return bestValidLoss;
		}

	private:
		System m_system;
		std::vector<State> m_trainStates;
		std::vector<State> m_validStates;
		static std::mutex m_initMutex;
		static std::mutex m_loggingMutex;
	};

	template<typename Network, typename System, typename Integrator>
	std::mutex TrainNetwork<Network, System, Integrator>::m_initMutex;

	template<typename Network, typename System, typename Integrator>
	std::mutex TrainNetwork<Network, System, Integrator>::m_loggingMutex;
}
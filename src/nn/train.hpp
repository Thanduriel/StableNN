#pragma once

#include "nnmaker.hpp"
#include "utils.hpp"
#include "../defs.hpp"
#include "../generator.hpp"
#include "hyperparam.hpp"
#include "lrscheduler.hpp"
#include <torch/torch.h>
#include <mutex>
#include <chrono>

namespace nn {

	template<typename Network, typename System, typename Integrator, typename InputMaker = MakeTensor_t<Network>>
	struct TrainNetwork
	{
		using State = typename System::State;
		using ValueT = typename System::ValueT;

		TrainNetwork(const System& _system, std::vector<State> _trainStates, std::vector<State> _validStates, std::vector<size_t> _warmupSteps = {})
			: TrainNetwork(std::vector{ _system }, std::move(_trainStates), std::move(_validStates), std::move(_warmupSteps))
		{
		}

		TrainNetwork(std::vector<System> _systems, std::vector<State> _trainStates, std::vector<State> _validStates, 
			std::vector<size_t> _warmupSteps = {})
			: m_systems(std::move(_systems)),
			m_trainStates(std::move(_trainStates)),
			m_validStates(std::move(_validStates)),
			m_warmupSteps(std::move(_warmupSteps))
		{
		}

		double operator()(const nn::HyperParams& _params) const
		{
			int64_t hyperSampleRate = *_params.get<int>("hyper_sample_rate");
			// system is just a placeholder
			auto makeIntegrator = [&]()
			{
				// integrator implements temporal hyper sampling already
				if constexpr (std::is_constructible_v<Integrator, System, ValueT, State, int>)
				{
					auto integ = Integrator(m_systems[0], *_params.get<double>("time_step"), State{}, hyperSampleRate);
					hyperSampleRate = 1;
					return integ;
				}
				else
					return Integrator(m_systems[0], *_params.get<double>("time_step") / hyperSampleRate);
			};
			Integrator referenceIntegrator = makeIntegrator();
			
			// distribute systems to training and validation generation
			const size_t numTotalStates = m_validStates.size() + m_trainStates.size();
			const size_t numTrainSystems = std::max(static_cast<size_t>(1), (m_trainStates.size() * m_systems.size()) / numTotalStates );
			const size_t numValidSystems = std::max(static_cast<size_t>(1), m_systems.size() - numTrainSystems);
			
			DataGenerator<System, Integrator, InputMaker> trainGenerator(
				std::vector<System>(m_systems.begin(), m_systems.begin() + numTrainSystems), 
				referenceIntegrator);
			DataGenerator<System, Integrator, InputMaker> validGenerator(
				std::vector<System>(m_systems.end() - numValidSystems, m_systems.end()),
				referenceIntegrator);

			auto start = std::chrono::high_resolution_clock::now();
			namespace dat = torch::data;
			const size_t numInputs = _params.get<size_t>("num_inputs", NUM_INPUTS);
			auto dataset = trainGenerator.generate(m_trainStates, *_params.get<int>("train_samples"), hyperSampleRate, numInputs, USE_SINGLE_OUTPUT, NUM_FORWARDS, m_warmupSteps)
				.map(dat::transforms::Stack<>());
			auto validationSet = validGenerator.generate(m_validStates, *_params.get<int>("valid_samples"), hyperSampleRate, numInputs, USE_SINGLE_OUTPUT, NUM_FORWARDS, m_warmupSteps)
				.map(dat::transforms::Stack<>());

			auto end = std::chrono::high_resolution_clock::now();
			const float genTime = std::chrono::duration<float>(end - start).count();
			// not constexpr so that genTime is used
			if (MODE != Mode::TRAIN_MULTI)
				std::cout << "Generating data took " << genTime << "s\n";

			// LBFGS does not work with mini batches and random sampling
			using Sampler = std::conditional_t<USE_SEQ_SAMPLER,
				dat::samplers::SequentialSampler,
				dat::samplers::RandomSampler>;
			auto data_loader = dat::make_data_loader<Sampler>(
				dataset,
				dat::DataLoaderOptions().batch_size(USE_LBFGS ? std::numeric_limits< size_t>::max() : _params.get<int>("batch_size", 64)));
			// make validation as fast as possible
			auto validationLoader = dat::make_data_loader<dat::samplers::SequentialSampler>(
				validationSet,
				dat::DataLoaderOptions().batch_size(std::numeric_limits< size_t>::max()));

			if constexpr (THREAD_FIXED_SEED)
			{
				s_initMutex.lock();
				torch::manual_seed(TORCH_SEED);
			}
			constexpr int stateSize = systems::sizeOfState<System>();
			auto net = nn::makeNetwork<Network, USE_WRAPPER, stateSize>(_params);
			auto bestNet = nn::makeNetwork<Network, USE_WRAPPER, stateSize>(_params);
			if constexpr (THREAD_FIXED_SEED)
			{
				s_initMutex.unlock();
			}

			const int loss_p = _params.get("loss_p", 2);
			auto lossFn = [loss_p](const torch::Tensor& self, const torch::Tensor& target)
			{
				return nn::lp_loss(self, target, loss_p);
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
				if constexpr (OPTIMIZER == Optimizer::LBFGS)
					return torch::optim::LBFGS(net->parameters(),
						torch::optim::LBFGSOptions(*_params.get<double>("lr")));
				else if constexpr (OPTIMIZER == Optimizer::ADAM)
					return torch::optim::Adam(net->parameters(),
						torch::optim::AdamOptions(_params.get<double>("lr", 3.e-4))
						.weight_decay(_params.get<double>("weight_decay", 1.e-6))
						.amsgrad(_params.get<bool>("amsgrad", false)));
				else if constexpr (OPTIMIZER == Optimizer::SGD)
					return torch::optim::SGD(net->parameters(),
						torch::optim::SGDOptions(_params.get<double>("lr", 0.01))
						.weight_decay(_params.get<double>("weight_decay", 1.e-6))
						.momentum(_params.get<double>("momentum", 0.0))
						.dampening(_params.get<double>("dampening", 0.0)));
				else if constexpr (OPTIMIZER == Optimizer::RMSPROP)
					return torch::optim::RMSprop(net->parameters(),
						torch::optim::RMSpropOptions(_params.get<double>("lr", 0.001))
						.momentum(_params.get<double>("momentum", 0.9)));
			};
			auto optimizer = makeOptimizer();
			auto lrScheduler = LearningRateScheduler(optimizer, _params.get<double>("lr_decay", 1.0));

			double bestValidLoss = std::numeric_limits<double>::max();

			//std::ofstream lossFile("loss.txt");

			const int64_t numEpochs = _params.get<int>("num_epochs", 2048);
			for (int64_t epoch = 1; epoch <= numEpochs; ++epoch)
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
				lrScheduler.step();

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

				if constexpr (MODE != Mode::TRAIN_MULTI)
				{
					if (epoch % 16 == 0)
					{
						constexpr int intervals = 20;
						const int progress = static_cast<int>(static_cast<float>(epoch * intervals) / numEpochs);
						std::cout << "<";
						for (int k = 0; k < progress; ++k)
							std::cout << "#";
						for (int k = progress; k < intervals; ++k)
							std::cout << " ";
						std::cout << "> [" << epoch << "/" << numEpochs << "] train loss: " << totalLoss.item<double>() << "\n";
					}
				}
			}
			if constexpr(LOG_LOSS)
			{
				std::unique_lock<std::mutex> lock(s_loggingMutex);
				std::ofstream lossLog("losses.txt", std::ios::app);
				lossLog << bestValidLoss << std::endl;
			}
			if constexpr(SAVE_NET)
			{
				//torch::save(bestNet, _params.get<std::string>("name", "net") + ".pt");
				nn::save(bestNet, _params);
			}

			return bestValidLoss;
		}

	private:
		std::vector<System> m_systems;
		std::vector<State> m_trainStates;
		std::vector<State> m_validStates;
		std::vector<size_t> m_warmupSteps;
		static std::mutex s_initMutex;
		static std::mutex s_loggingMutex;
	};

	template<typename Network, typename System, typename Integrator, typename InputMaker>
	std::mutex TrainNetwork<Network, System, Integrator, InputMaker>::s_initMutex;

	template<typename Network, typename System, typename Integrator, typename InputMaker>
	std::mutex TrainNetwork<Network, System, Integrator, InputMaker>::s_loggingMutex;
}
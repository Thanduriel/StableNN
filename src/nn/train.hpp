#pragma once

#include "nnmaker.hpp"
#include "utils.hpp"
#include "../defs.hpp"
#include "../generator.hpp"
#include "hyperparam.hpp"
#include "lrscheduler.hpp"
#include "loss.hpp"
#include <torch/torch.h>
#include <mutex>
#include <chrono>

namespace nn {

	// Functor that handles data generation, network creation and training. 
	// @param UseWrapper Use a wrapper network to increase the number of inputs or reduce the number of outputs.
	//					 Currently does not work with convolutional networks,
	template<typename Network, 
		typename System, 
		typename Integrator, 
		typename InputMaker = MakeTensor_t<Network>, 
		typename OutputMaker = StateToTensor,
		bool UseWrapper = true>
	struct TrainNetwork
	{
		using State = typename System::State;
		using ValueT = typename System::ValueT;

		TrainNetwork(const System& _system, 
			std::vector<State> _trainStates, 
			std::vector<State> _validStates,
			std::vector<size_t> _warmupSteps = {})
			: TrainNetwork(std::vector{ _system }, 
				std::vector{ _system }, 
				std::move(_trainStates), 
				std::move(_validStates), 
				std::move(_warmupSteps))
		{
		}

		TrainNetwork(std::vector<System> _trainSystems, 
			std::vector<System> _validSystems,
			std::vector<State> _trainStates, std::vector<State> _validStates, 
			std::vector<size_t> _warmupSteps = {})
			: m_trainSystems(std::move(_trainSystems)),
			m_validSystems(std::move(_validSystems)),
			m_trainStates(std::move(_trainStates)),
			m_validStates(std::move(_validStates)),
			m_warmupSteps(std::move(_warmupSteps))
		{
		}

		// Run training with hyper params _params.
		double operator()(const nn::HyperParams& _params) const
		{
			namespace dat = torch::data;

			int64_t hyperSampleRate = *_params.get<int>("hyper_sample_rate");
			// system is just a placeholder
			auto makeIntegrator = [&]()
			{
				// integrator implements temporal hyper sampling already
				if constexpr (std::is_constructible_v<Integrator, System, ValueT, State, int>)
				{
					auto integ = Integrator(m_trainSystems[0], *_params.get<double>("time_step"), State{}, hyperSampleRate);
					hyperSampleRate = 1;
					return integ;
				}
				else
					return Integrator(m_trainSystems[0], *_params.get<double>("time_step") / hyperSampleRate);
			};
			Integrator referenceIntegrator = makeIntegrator();
			
			DataGenerator<System, Integrator, InputMaker, OutputMaker> trainGenerator(m_trainSystems, referenceIntegrator);
			DataGenerator<System, Integrator, InputMaker, OutputMaker> validGenerator(m_validSystems, referenceIntegrator);

			auto start = std::chrono::high_resolution_clock::now();
			const size_t numInputs = _params.get<size_t>("num_inputs", NUM_INPUTS);
			auto dataset = trainGenerator.generate(m_trainStates, *_params.get<int>("train_samples"), hyperSampleRate, numInputs, USE_SINGLE_OUTPUT, NUM_FORWARDS, m_warmupSteps)
				.map(dat::transforms::Stack<>());
			auto validationSet = validGenerator.generate(m_validStates, *_params.get<int>("valid_samples"), hyperSampleRate, numInputs, USE_SINGLE_OUTPUT, NUM_FORWARDS, m_warmupSteps)
				.map(dat::transforms::Stack<>());

			auto end = std::chrono::high_resolution_clock::now();
			const float genTime = std::chrono::duration<float>(end - start).count();
			// not constexpr to prevent warnings
			if (MODE != Mode::TRAIN_MULTI)
				std::cout << "Generating data took " << genTime << "s\n";
			
			torch::Device device(torch::kCPU);
			if (torch::cuda::is_available() && _params.get<bool>("train_gpu", false)) 
				device = torch::kCUDA;

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

			// enforce that network initialization uses the given seed
			if constexpr (THREAD_FIXED_SEED)
			{
				s_initMutex.lock();
				torch::manual_seed(_params.get<uint64_t>("seed", TORCH_SEED));
			}
			const bool loadNet = _params.get<bool>("load_net", false);
			HyperParams loadedParams = _params;
			auto net = loadNet ? nn::load<Network, UseWrapper>(_params, "", device, &loadedParams)
				: nn::makeNetwork<Network, UseWrapper>(_params, device);
			auto bestNet = nn::clone(net);

			if constexpr (THREAD_FIXED_SEED)
			{
				s_initMutex.unlock();
			}
			if constexpr (MODE != Mode::TRAIN_MULTI)
				std::cout << "Training network with " << nn::countParams(net) << " parameters.\n";

			// construct loss function
			auto lossFnTrain = makeLossFunction(_params, true);
			auto lossFnValid = makeLossFunction(_params, false);

			// input preprocessing for multi-step forward mode
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
						torch::optim::LBFGSOptions(*_params.get<double>("lr"))
						.history_size(_params.get<int>("history_size", 100)));
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
			auto lrScheduler = LearningRateScheduler(optimizer, _params.get<double>("lr_decay", 1.0), _params.get<int64_t>("lr_epoch_update", 1));

			std::ofstream learnLossLog = LOG_LEARNING_LOSS ? std::ofstream(_params.get<std::string>("name", "") + "_loss.txt") : std::ofstream();
			double bestValidLoss = std::numeric_limits<double>::max();

			start = std::chrono::high_resolution_clock::now();
			const int64_t numEpochs = _params.get<int>("num_epochs", 2048);
			for (int64_t epoch = 1; epoch <= numEpochs; ++epoch)
			{
				// train
				net->train();

				double totalLoss = 0.0;
				int forwardRuns = 0;

				for (dat::Example<>& batch : *data_loader)
				{
					torch::Tensor data = batch.data.to(device);
					torch::Tensor target = batch.target.to(device);

					auto closure = [&]()
					{
						net->zero_grad();
						torch::Tensor output;
						torch::Tensor input = data.clone();
						for (int64_t i = 0; i < NUM_FORWARDS; ++i)
						{
							output = net->forward(input);
							input = nextInput(input, output);
						}
						torch::Tensor loss = lossFnTrain(output, target, IdentityMap<Network>::forward(data));
						totalLoss += loss.item<double>();
						++forwardRuns; // count runs to normalize the training error

						loss.backward();
						return loss;
					};

					optimizer.step(closure);
				}
				lrScheduler.step(epoch);
				const double trainLoss = totalLoss / forwardRuns;

				// validation
				torch::NoGradGuard gradGuard;
				net->eval();
				double validLoss = 0.0;
				for (dat::Example<>& batch : *validationLoader)
				{
					torch::Tensor input = batch.data.to(device);
					torch::Tensor target = batch.target.to(device);
					torch::Tensor output;

					for (int64_t i = 0; i < NUM_FORWARDS; ++i)
					{
						output = net->forward(input);
						input = nextInput(input, output);
					}
					// last argument should not be used by the validation loss
					torch::Tensor loss = lossFnValid(output, target, input);
					validLoss += loss.item<double>();
				}

				if constexpr (LOG_LEARNING_LOSS)
				{
					learnLossLog << trainLoss
						<< ", " << validLoss << "\n";
				}

				if (validLoss < bestValidLoss)
				{
					if constexpr (MODE != Mode::TRAIN_MULTI)
						std::cout << validLoss << "\n";
					bestNet = nn::clone(net);
					bestValidLoss = validLoss;
				}

				if constexpr (MODE != Mode::TRAIN_MULTI)
				{
					if (epoch % 16 == 0)
					{
						constexpr int INTERVALS = 20;
						const int progress = static_cast<int>(static_cast<float>(epoch * INTERVALS) / numEpochs);
						std::cout << "<";
						for (int k = 0; k < progress; ++k)
							std::cout << "#";
						for (int k = progress; k < INTERVALS; ++k)
							std::cout << " ";
						std::cout << "> [" << epoch << "/" << numEpochs << "] train loss: " << trainLoss << "\n";
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
				// store loaded params again to ensure that at least the network parameters are correct
				if (loadNet)
					loadedParams["load_net"] = true;
				nn::save(bestNet, loadedParams);
			}
			end = std::chrono::high_resolution_clock::now();
			const float trainTime = std::chrono::duration<float>(end - start).count();
			// not constexpr to prevent warnings
			if (MODE != Mode::TRAIN_MULTI)
			{
				std::cout << "Finished training in " << trainTime
					<< "s. The final validation loss is " << bestValidLoss << ".\n";
			}

			return bestValidLoss;
		}

	private:
		std::vector<System> m_trainSystems;
		std::vector<System> m_validSystems;
		std::vector<State> m_trainStates;
		std::vector<State> m_validStates;
		std::vector<size_t> m_warmupSteps;
		static std::mutex s_initMutex;
		static std::mutex s_loggingMutex;
	};

	template<typename Network, typename System, typename Integrator, typename InputMaker, typename OutputMaker, bool UseWrapper>
	std::mutex TrainNetwork<Network, System, Integrator, InputMaker, OutputMaker, UseWrapper>::s_initMutex;

	template<typename Network, typename System, typename Integrator, typename InputMaker, typename OutputMaker, bool UseWrapper>
	std::mutex TrainNetwork<Network, System, Integrator, InputMaker, OutputMaker, UseWrapper>::s_loggingMutex;
}
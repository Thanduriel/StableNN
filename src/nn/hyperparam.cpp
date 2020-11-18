#include "hyperparam.hpp"
#include <random>
#include <iostream>
#include <thread>
#include <mutex>


namespace nn {

/*	template<template<typename T, typename Types>
	std::ostream& operator<<(std::ostream& _out, const std::any& _value);
	{

	}*/

	template<typename Dummy, typename T, typename... Types>
	static void serialize(std::ostream& _out, const std::any& _val)
	{
		if (_val.type() == typeid(T))
			_out << std::any_cast<T>(_val);
		else
			serialize<Dummy, Types...>(_out, _val);
	}

	template<typename Dummy>
	static void serialize(std::ostream& _out, const std::any& _val)
	{
		_out << "Unknown type " << _val.type().name();
	}


	std::ostream& operator<<(std::ostream& _out, const HyperParams& _params)
	{
		std::cout << "{";
		for (const auto& [name, value] : _params.data)
		{
			_out << name << " : ";
			serialize<void, int, double, float, int32_t, int64_t, uint32_t, size_t, std::string, bool>(_out, value);
			std::cout << ", ";
		}
		std::cout << "}";

		return _out;
	}

	std::pair<size_t, size_t> GridSearchOptimizer::decomposeFlatIndex(size_t flatIndex, int _k) const
	{
		size_t reminder = flatIndex;

		size_t flatInd = 0;
		size_t dimSize = 1;
		for (int j = 0; j < _k; ++j)
		{
			const size_t sizeJ = m_hyperGrid[j].second.size();
			size_t indJ = reminder % sizeJ;
			reminder /= sizeJ;

			flatInd += dimSize * indJ;
			dimSize *= sizeJ;
		}

		const size_t sizeK = m_hyperGrid[_k].second.size();
		const size_t indK = reminder % sizeK;
		reminder /= sizeK;

		flatInd += reminder * dimSize;

		return { indK, flatInd };
	}

	void GridSearchOptimizer::run(unsigned numThreads)
	{
		size_t numOptions = 1;
		for (const auto& [name, values] : m_hyperGrid) 
		{
			numOptions *= values.size();
		}
		std::vector<double> results(numOptions);
		size_t bestResult = 0;
		HyperParams bestParams;
		std::mutex mutex;

		auto work = [&](size_t begin, size_t end) 
		{
			for (size_t i = begin; i < end; ++i)
			{
				HyperParams params;
				std::string fileName = "";

				// setup param file
				size_t remainder = i;
				for (const auto& [name, values] : m_hyperGrid)
				{
					const size_t ind = remainder % values.size();
					remainder /= values.size();
					params[name] = values[ind];
					fileName += std::to_string(ind) + "_";
				}
				params["name"] = fileName + ".pt";

				const double loss = m_trainFunc(params);
				results[i] = loss;

				std::lock_guard<std::mutex> guard(mutex);
				std::cout << "training with " << params << std::endl;
				std::cout << "loss: " << loss << std::endl;

				if (loss < results[bestResult])
				{
					bestResult = i;
					bestParams = params;
				}
			}
		};
		if (numThreads == 1)
			work(0, numOptions);
		else
		{
			std::vector<std::thread> threads;
			const size_t numRuns = numOptions / numThreads;
			for (unsigned i = 0; i < numThreads - 1; ++i)
				threads.emplace_back(work, i * numRuns, (i + 1) * numRuns);
			work((numThreads - 1) * numRuns, numOptions);
			for (auto& t : threads)
				t.join();
		}
		
		
		// print results

		// combined results over each parameter
		for (size_t k = 0; k < m_hyperGrid.size(); ++k)
		{
			const auto& [name, values] = m_hyperGrid[k];
			std::vector<double> losses(values.size(), 0.0);

			for (size_t i = 0; i < numOptions; ++i)
			{
				const auto& [indK, indOth] = decomposeFlatIndex(i, k);
				losses[indK] += results[i];
			}
			
			std::cout << name << ": ";
			for (double loss : losses)
				std::cout << loss << ", ";
			std::cout << "\n";
		}

		std::cout << "\n============================\n best result:\n" << bestParams << "\n"
			<< "loss: " << results[bestResult] << std::endl;
	/*	size_t remainder = bestResult;
		for (const auto& [name, values] : m_hyperGrid) 
		{
			const size_t ind = remainder % values.size();
			remainder /= values.size();
			std::cout << name << ": " << ind;

			// if possible also print the value itself
			const std::any& val = values[ind];
			if (val.type() == typeid(float))
				std::cout << " (" << std::any_cast<float>(val) << ")";
			else if (val.type() == typeid(double))
				std::cout << " (" << std::any_cast<double>(val) << ")";

			std::cout << "\n";
		}*/
	}

	RandomSearchOptimizer::RandomSearchOptimizer(const TrainFn& _trainFn, uint32_t _seed)
		: m_trainFn(_trainFn),
		m_rng(_seed)
	{}

	void RandomSearchOptimizer::run(int _tries)
	{
		HyperParams bestParams;
		double bestLoss = std::numeric_limits<double>::max();

		for (int i = 0; i < _tries; ++i)
		{
			HyperParams params;
			for (auto& [name, sampler] : m_paramSamplers)
			{
				params[name] = sampler(m_rng);
			}

			const double loss = m_trainFn(params);
			std::cout << params << "\nwith loss: " << loss << std::endl;
			if (loss < bestLoss)
			{
				bestLoss = loss;
				bestParams = params;
			}
		}

		std::cout << "\n============================\n best result:\n" << bestParams << "\n"
			<< "loss: " << bestLoss << std::endl;
	}

/*
void HyperParamOptimizer::run(TrainFn train_fn){
  
  std::default_random_engine rng(0xB123AF5E);

  for(;;){
	// sample params
	HyperParams params;
	for(const auto& [name, values] : m_hyperGrid){
	  std::uniform_int_distribution<size_t> dist(0, values.size());
	  params[name] = values[dist(rng)];
	}

	const double loss = train_fn(params);
  }
}*/



}
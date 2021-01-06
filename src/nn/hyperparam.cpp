#include "hyperparam.hpp"
#include <random>
#include <iostream>
#include <thread>
#include <mutex>
#include <algorithm>


namespace nn {

	template<typename Dummy, typename T, typename... Types>
	static void serialize(std::ostream& _out, const std::any& _val)
	{
		if (_val.type() == typeid(T))
			_out << std::any_cast<T>(_val);
		else
		{
			if constexpr (static_cast<bool>(sizeof...(Types)))
				serialize<Dummy, Types...>(_out, _val);
			else
				_out << "Unknown type " << _val.type().name();
		}
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

	void GridSearchOptimizer::run(unsigned numThreads) const
	{
		size_t numOptions = 1;
		for (const auto& [name, values] : m_hyperGrid) 
		{
			numOptions *= values.size();
		}
		std::vector<double> results(numOptions);
		double bestResult = std::numeric_limits<double>::max();
		HyperParams bestParams;
		std::mutex mutex;

		auto work = [&mutex, &bestResult, &bestParams, &results, this](size_t begin, size_t end) 
		{
			for (size_t i = begin; i < end; ++i)
			{
				HyperParams params(m_defaultParams);
				HyperParams changedParams;
				std::string fileName = "";

				// setup param file
				size_t remainder = i;
				for (const auto& [name, values] : m_hyperGrid)
				{
					const size_t ind = remainder % values.size();
					remainder /= values.size();
					params[name] = values[ind];
					changedParams[name] = values[ind];
					fileName += std::to_string(ind) + "_";
				}
				params["name"] = fileName + m_defaultParams.get<std::string>("name", "") +".pt";

				const double loss = m_trainFunc(params);
				results[i] = loss;

				const std::lock_guard<std::mutex> guard(mutex);
				std::cout << "training with " << changedParams << "\n";
				std::cout << "loss: " << loss << std::endl;

				if (loss < bestResult)
				{
					bestResult = loss;
					bestParams = params;
				}
			}
		};
		if (numThreads == 1)
			work(0, numOptions);
		else
		{
			std::vector<std::thread> threads;
			numThreads = std::min(static_cast<unsigned>(numOptions), numThreads);
			const size_t numRuns = numOptions / numThreads;
			for (unsigned i = 0; i < numThreads - 1; ++i)
				threads.emplace_back(work, i * numRuns, (i + 1) * numRuns);
			work((numThreads - 1) * numRuns, numOptions);
			for (auto& t : threads)
				t.join();
		}
		
		
		// print results

		// combined results over each parameter
		std::cout << "\nAverage loss over parameters:\n";
		for (size_t k = 0; k < m_hyperGrid.size(); ++k)
		{
			const auto& [name, values] = m_hyperGrid[k];
			std::vector<double> losses(values.size(), 0.0);
			std::vector<int> sizes(values.size(), 0);

			for (size_t i = 0; i < numOptions; ++i)
			{
				const auto& [indK, indOth] = decomposeFlatIndex(i, k);
				losses[indK] += results[i];
				sizes[indK]++;
			}
			
			std::cout << name << ": ";
			for (size_t i = 0; i < losses.size(); ++i)
				std::cout << losses[i] / sizes[i] << ", ";
			std::cout << "\n";
		}

		// results for parameter pairs
		std::cout << "\nAverage loss over parameter pairs:\n";
		for (size_t k1 = 0; k1 < m_hyperGrid.size(); ++k1)
		{
			for (size_t k2 = k1 + 1; k2 < m_hyperGrid.size(); ++k2)
			{
				const auto& [name1, values1] = m_hyperGrid[k1];
				const auto& [name2, values2] = m_hyperGrid[k2];
				const size_t size1 = values1.size();
				const size_t size2 = values2.size();
				const size_t size = size1 * size2;
				std::vector<double> losses(size, 0.0);
				std::vector<int> sizes(size, 0);

				for (size_t i = 0; i < numOptions; ++i)
				{
					const auto& [indK1, _] = decomposeFlatIndex(i, k1);
					const auto& [indK2, __] = decomposeFlatIndex(i, k2);
					const size_t idx = indK1 * size2 + indK2;
					losses[idx] += results[i];
					sizes[idx]++;
				}

				std::cout << "\n" << name1 << " \\ " << name2 << "\n";
				printf("%4.d", 0);
				for (int j = 0; j < static_cast<int>(size2); ++j)
					printf("%11d ", j);
				printf("\n");
				for (size_t i = 0; i < size1; ++i)
				{
					printf("%3d ", static_cast<int>(i));
					const size_t indPart = i * size2;
					for (size_t j = 0; j < size2; ++j)
					{
						const size_t idx = indPart + j;
						printf("%.5e ", losses[idx] / sizes[idx]);
					}
					printf("\n");
				}
			}
			std::cout << "\n";
		}

		std::cout << "\n============================\n best result:\n" << bestParams << "\n"
			<< "loss: " << bestResult << std::endl;
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



}
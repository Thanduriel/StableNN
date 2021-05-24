#include "hyperparam.hpp"
#include <random>
#include <iostream>
#include <thread>
#include <mutex>
#include <algorithm>
#include <chrono>
#include <sstream>


namespace nn {
	std::string sanitizeString(const std::string& _str)
	{
		std::string s = _str;
		std::replace(s.begin(), s.end(), ' ', '_'); // breaks >>
		std::replace(s.begin(), s.end(), ',', ';'); // needed to differentiate keys
		return s;
	}

	void ExtAny::print(std::ostream& _out) const
	{
		m_print(_out, any);
	}

	void ExtAny::read(std::istream& _stream)
	{
		m_read(_stream, any);
	}

	std::ostream& operator<<(std::ostream& _out, const HyperParams& _params)
	{
		_out << "{ ";
		for (const auto& [name, value] : _params.m_data)
		{
			_out << name << " : ";
			value.print(_out);
			_out << ", ";
		}
		_out << "}";

		return _out;
	}



	std::istream& operator>>(std::istream& _in, HyperParams& _params)
	{
		std::string buf;
		_in >> buf; // "{"
		while (_in.good())
		{
			getline(_in, buf, ',');
			std::stringstream entry(buf);

			std::string key;
			entry >> key;
			if (key == "}") break;
			entry >> buf; // ":"
			auto it = _params.m_data.find(key);
			if (it == _params.m_data.end())
				std::cerr << "[Warning] Encountered unknown key while parsing param file: \"" << key << "\".\n";
			else
				it->second.read(entry);
		}

		return _in;
	}

	GridSearchOptimizer::GridSearchOptimizer(const TrainFn& _trainFn, const HyperParamGrid& _paramGrid,
		const HyperParams& _defaults)
		: m_hyperGrid(_paramGrid), m_trainFunc(_trainFn), m_defaultParams(_defaults)
	{}

	HyperParams GridSearchOptimizer::run(unsigned numThreads) const
	{
		// compute number of configurations
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
				params["name"] = fileName + m_defaultParams.get<std::string>("name", "");

				auto start = std::chrono::high_resolution_clock::now();
				const double loss = m_trainFunc(params);
				auto end = std::chrono::high_resolution_clock::now();
				const float time = std::chrono::duration<float>(end - start).count();
				results[i] = loss;

				const std::lock_guard<std::mutex> guard(mutex);
				std::cout << "training with " << changedParams << "\n";
				std::cout << "loss: " << loss << "     time: " << time << "s" << std::endl;

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
		std::cout << "\n================== Evaluation ==================\n";
		double avg = 0.0;
		for (double d : results)
			avg += d;
		avg /= results.size();
		std::cout << "\nAverage loss: " << avg << "\n";
		// combined results over each parameter
		std::cout << "Average loss over parameters:\n";
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

			double variance = 0.0;
			for (double d : losses)
				variance += (avg - d) * (avg - d);
			variance /= losses.size();
			
			std::cout << name << ": ";
			for (size_t i = 0; i < values.size(); ++i)
			{
				values[i].print(std::cout); 
				std::cout << ", ";
			}
			std::cout << "\n    ";
			for (size_t i = 0; i < losses.size(); ++i)
				std::cout << losses[i] / sizes[i] << ", ";
			std::cout << "[std deviation: " << std::sqrt(variance) << "]\n";
		}

		// result matrices for parameter pairs
		if (m_hyperGrid.size() > 1)
		{
			std::cout << "\nAverage loss over parameter pairs:";
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
			}
		}

		std::cout << "\nbest result:\n" << bestParams << "\n"
			<< "loss: " << bestResult << std::endl;

		bestParams["validation_loss"] = bestResult;
		return bestParams;
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
}
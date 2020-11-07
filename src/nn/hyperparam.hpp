#pragma once

#include <any>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <random>

namespace nn {

	class HyperParams
	{
	public:
		std::any& operator[](const std::string& key) { return data[key]; }

		template<typename T>
		T get(const std::string& key, const T& default_value) const
		{
			const auto it = data.find(key);
			return it != data.end() ? std::any_cast<T>(it->second) : default_value;
		}

		friend std::ostream& operator<<(std::ostream& _out, const HyperParams& _params);

		// enable foreach loop
		auto begin() { return data.begin(); }
		auto end() { return data.end(); }
	private:
		std::unordered_map<std::string, std::any> data;
	};

	using HyperParamGrid = std::vector<std::pair<std::string, std::vector<std::any>>>;
	using TrainFn = std::function<double(const HyperParams&)>;

	class GridSearchOptimizer
	{
	public:
		GridSearchOptimizer(const TrainFn& _trainFn, const HyperParamGrid& _paramGrid)
			: m_hyperGrid(_paramGrid), m_trainFunc(_trainFn)
		{}

		void run();
	private:
		HyperParamGrid m_hyperGrid;
		TrainFn m_trainFunc;
	};


	class RandomSearchOptimizer
	{
	public:
		using RandomEngine = std::default_random_engine;

		RandomSearchOptimizer(const TrainFn& _trainFn, uint32_t _seed);

		template<typename Sampler>
		void addParam(const std::string& _name, Sampler _sampler)
		{
			m_paramSamplers.emplace_back(_name,[=](RandomEngine& _rng)
				{
					return { _sampler(_rng)};
				});
		}

		void run(int _tries);
	private:
		std::vector<std::pair<std::string, std::function<std::any(RandomEngine& _rng)>>> m_paramSamplers;
		TrainFn m_trainFn;
		std::default_random_engine m_rng;
	};

}

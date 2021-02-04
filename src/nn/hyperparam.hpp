#pragma once

#include <any>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <random>
#include <optional>

namespace nn {

	class HyperParams
	{
	public:
		std::any& operator[](const std::string& key) { return data[key]; }

		template<typename T>
		T get(const std::string& _key, const T& _defaultValue) const
		{
			const auto it = data.find(_key);
			if (std::optional<T> opt; it != data.end() && (opt = cast<T>(it->second)).has_value())
			{
				return *opt;
			}
			return _defaultValue;
		}

		template<typename T>
		std::optional<T> get(const std::string& _key) const
		{
			const auto it = data.find(_key);
			if (it != data.end())
				return cast<T>(it->second);
			else return {};
		}

		friend std::ostream& operator<<(std::ostream& _out, const HyperParams& _params);

		// enable foreach loop
		auto begin() { return data.begin(); }
		auto end() { return data.end(); }
	private:
		// determine whether tuple contains a specific type
		template <typename T, typename Tuple>
		struct has_type;

		template <typename T, typename... Us>
		struct has_type<T, std::tuple<Us...>> : std::disjunction<std::is_same<T, Us>...> {};

		using INTEGRAL_TYPES = std::tuple<int, unsigned, int64_t, size_t, uint64_t>;

		template<typename T>
		static std::optional<T> cast(const std::any& any)
		{
			if(any.type() == typeid(T))
				return std::any_cast<T>(any);

			// integral types are somewhat interchangeable
			if constexpr (has_type<T, INTEGRAL_TYPES>::value)
			{
				return tryCastTuple<T>(any, INTEGRAL_TYPES{});
			}

			return std::nullopt;
		}

		template<typename T, typename... Us>
		static std::optional<T> tryCastTuple(const std::any& any, std::tuple<Us...>)
		{
			return tryCastTuple<T, Us...>(any);
		}

		template<typename T, typename U, typename... Us>
		static std::optional<T> tryCastTuple(const std::any& any)
		{
			if (any.type() == typeid(U))
			{
				const U u = std::any_cast<U>(any);
				const T t = static_cast<T>(u);
				if (static_cast<U>(t) != u) // check that no information is lost
					std::cout << "[Warning] Stored parameter " << u << " was cast to " << t << std::endl;
				return t;
			}

			if constexpr (sizeof...(Us))
				return tryCastTuple<T, Us...>(any);
			else
				return std::nullopt;
		}

		std::unordered_map<std::string, std::any> data;
	};

	using HyperParamGrid = std::vector<std::pair<std::string, std::vector<std::any>>>;
	using TrainFn = std::function<double(const HyperParams&)>;

	class GridSearchOptimizer
	{
	public:
		GridSearchOptimizer(const TrainFn& _trainFn, const HyperParamGrid& _paramGrid,
			const HyperParams& _defaults = {});

		void run(unsigned _numThreads = 1) const;
	private:
		std::pair<size_t, size_t> decomposeFlatIndex(size_t _flatIndex, int _k) const;

		HyperParamGrid m_hyperGrid;
		TrainFn m_trainFunc;
		HyperParams m_defaultParams;
	};


	class RandomSearchOptimizer
	{
	public:
		using RandomEngine = std::default_random_engine;

		RandomSearchOptimizer(const TrainFn& _trainFn, uint32_t _seed);

		template<typename Sampler>
		void addParam(const std::string& _name, Sampler _sampler)
		{
		/*	m_paramSamplers.emplace_back(_name,[=](RandomEngine& _rng)
				{
					return { _sampler(_rng)};
				});*/
		}

		void run(int _tries);
	private:
		std::vector<std::pair<std::string, std::function<std::any(RandomEngine& _rng)>>> m_paramSamplers;
		TrainFn m_trainFn;
		std::default_random_engine m_rng;
	};

}

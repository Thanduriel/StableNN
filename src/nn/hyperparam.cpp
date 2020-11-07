#include "hyperparam.hpp"
#include <random>
#include <iostream>


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
		_out << "Could not serialize value of type " << _val.type().name();
	}


	std::ostream& operator<<(std::ostream& _out, const HyperParams& _params)
	{
		for (const auto& [name, value] : _params.data)
		{
			_out << name;
			serialize<void, int, double, float, int32_t, int64_t, uint32_t>(_out, value);
			std::cout << "\n";
		}
	}

	void GridSearchOptimizer::run()
	{
		size_t numOptions = 1;
		for (const auto& [name, values] : m_hyperGrid) 
		{
			numOptions *= values.size();
		}
		std::vector<double> results(numOptions);
		size_t bestResult = 0;

		for (size_t i = 0; i < numOptions; ++i) 
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
			if (loss < results[bestResult])
				bestResult = i;
		}

		// print results
		size_t remainder = bestResult;
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
		}
	}

	RandomSearchOptimizer::RandomSearchOptimizer(const TrainFn& _trainFn, uint32_t _seed)
		: m_trainFn(_trainFn),
		m_rng(_seed)
	{}

	void RandomSearchOptimizer::run(int _tries)
	{
		for (int i = 0; i < _tries; ++i)
		{
			HyperParams params;
			for (auto& [name, sampler] : m_paramSamplers)
			{
				params[name] = sampler(m_rng);
			}

			const double loss = m_trainFn(params);
			std::cout << params;
		}
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
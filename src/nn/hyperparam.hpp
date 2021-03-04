#pragma once

#include <any>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <random>
#include <optional>
#include <exception>
#include <typeinfo>
#include <iostream>

namespace nn {

	namespace details {
		// type trait that checks for the operator<<(S&, T)
		template<typename S, typename T, typename = void>
		struct is_stream_writable : std::false_type {};
		template<typename S, typename T>
		struct is_stream_writable<S, T,
			std::void_t<  decltype(std::declval<S&>() << std::declval<T>())  > >
			: std::true_type {};

		// type trait that checks for the operator>>(S&, T&)
		template<typename S, typename T, typename = void>
		struct is_stream_readable : std::false_type {};
		template<typename S, typename T>
		struct is_stream_readable<S, T,
			std::void_t<  decltype(std::declval<S&>() >> std::declval<T&>())  > >
			: std::true_type {};

		template<typename T>
		void printImpl(std::ostream& _out, const std::any& _any)
		{
			if constexpr (is_stream_writable<std::ostream, T>::value)
				_out << std::any_cast<T>(_any);
			else
			{
				std::string name = _any.type().name();
				// sanitize string to not interfere with HyperParams serialization
				std::replace(name.begin(), name.end(), ',', ';');
				_out << name;
			}
		}

		template<typename T>
		void readImpl(std::istream& _in, std::any& _any)
		{
			if constexpr (is_stream_readable<std::istream, T>::value)
			{
				T val{};
				_in >> val;
				_any = val;
			}
			else
			{
				std::string s;
				_in >> s; // still consume the element
				std::cerr << "Could not read value of type " << _any.type().name() << "\n";
			}
		}
	}

	// Extended std::any which also stores type erased stream operators for serialization and deserialization.
	class ExtAny
	{
	public:
		std::any any;

		ExtAny() = default;
		ExtAny(const ExtAny& _oth) = default;
		ExtAny(ExtAny&& _oth) = default;

		// allow construction from any type besides ExtAny to not be confused with move construction
		template<typename T, std::enable_if_t<std::negation_v<std::is_same<std::decay_t<T>, ExtAny>>, int> = 0>
		ExtAny(T&& _val)
			: any(std::forward<T>(_val)),
			m_print(&details::printImpl<T>),
			m_read(&details::readImpl<std::decay_t<T>>)
		{
		}

		ExtAny& operator=(const ExtAny& _val) = default;
		ExtAny& operator=(ExtAny&& _val) = default;

		template<typename T, std::enable_if_t<std::negation_v<std::is_same<std::decay_t<T>, ExtAny>>, int> = 0>
		ExtAny& operator=(T&& _val)
		{
			any = std::forward<T>(_val);
			m_print = &details::printImpl<T>;
			m_read = &details::readImpl<std::decay_t<T>>;
			return *this;
		}

		// Works only when already initialized with a type.
		// No stream operators are provided since implicit conversion to ExtAny would enable this for all types.
		void print(std::ostream& _out) const;
		void read(std::istream& _out);

	private:
		using PrintFn = void(std::ostream&, const std::any&);
		using ReadFn = void(std::istream&, std::any&);

		PrintFn* m_print = nullptr;
		ReadFn* m_read = nullptr;
	};

	// Container which holds key value pairs with type erased values.
	class HyperParams
	{
	public:
		// Provides direct access to the underlying any. 
		// A new object is created if the key does not exist yet.
		ExtAny& operator[](const std::string& key) { return m_data[key]; }

		// Read a value of type T.
		// @return The associated value or _defaultValue if the key does not exist or its type is not compatible.
		template<typename T>
		T get(const std::string& _key, const T& _defaultValue) const
		{
			const auto it = m_data.find(_key);
			if (std::optional<T> opt; it != m_data.end() && (opt = cast<T>(it->second.any)).has_value())
			{
				return *opt;
			}
			return _defaultValue;
		}

		template<typename T>
		std::optional<T> get(const std::string& _key) const
		{
			const auto it = m_data.find(_key);
			if (it != m_data.end())
				return cast<T>(it->second.any);
			else return std::nullopt;
		}

		friend std::ostream& operator<<(std::ostream& _out, const HyperParams& _params);
		friend std::istream& operator>>(std::istream& _in, HyperParams& _params);

		// expose iterators to enable foreach-loop
		auto begin() { return m_data.begin(); }
		auto end() { return m_data.end(); }
		auto begin() const { return m_data.begin(); }
		auto end() const { return m_data.end(); }
	private:
		// determine whether tuple contains a specific type
		template <typename T, typename Tuple>
		struct has_type;
		template <typename T, typename... Us>
		struct has_type<T, std::tuple<Us...>> : std::disjunction<std::is_same<T, Us>...> {};

		// common types for implicit casts
		using INTEGRAL_TYPES = std::tuple<int, unsigned, int64_t, size_t, uint64_t>;
		using FLOATING_POINT_TYPES = std::tuple<float, double>;

		// less strict than std::any_cast allowing for some type conversions
		template<typename T>
		static std::optional<T> cast(const std::any& any)
		{
			if(any.type() == typeid(T))
				return std::any_cast<T>(any);

			// integral types are somewhat interchangeable
			if constexpr (has_type<T, INTEGRAL_TYPES>::value)
				return tryCastTuple<T>(any, INTEGRAL_TYPES{});
			if constexpr (has_type<T, FLOATING_POINT_TYPES>::value)
				return tryCastTuple<T>(any, FLOATING_POINT_TYPES{});

			return std::nullopt;
		}

		// try all types in the tuple
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
					throw std::bad_any_cast();
				return t;
			}

			if constexpr (sizeof...(Us))
				return tryCastTuple<T, Us...>(any);
			else
				return std::nullopt;
		}

		std::unordered_map<std::string, ExtAny> m_data;
	};

	// List of parameters to check with possible values.
	using HyperParamGrid = std::vector<std::pair<std::string, std::vector<ExtAny>>>;
	// A function which trains a network and returns the validation loss.
	using TrainFn = std::function<double(const HyperParams&)>;

	// Grid search optimizer for hyper parameters.
	class GridSearchOptimizer
	{
	public:
		// @param _defaults Set of parameters to use which are not set in _paramGrid.
		GridSearchOptimizer(const TrainFn& _trainFn, const HyperParamGrid& _paramGrid,
			const HyperParams& _defaults = {});

		// Run the grid search training up to _numThreads networks in parallel.
		// The number of actual threads can be higher if the network execution/training itself is multi-threaded.
		void run(unsigned _numThreads = 1) const;
	private:
		// Compute new indices for a k-flattening of the HyperParamGrid tensor from a flatIndex.
		std::pair<size_t, size_t> decomposeFlatIndex(size_t _flatIndex, int _k) const;

		HyperParamGrid m_hyperGrid;
		TrainFn m_trainFunc;
		HyperParams m_defaultParams;
	};

}

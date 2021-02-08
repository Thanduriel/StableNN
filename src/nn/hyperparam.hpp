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
		template<typename S, typename T, typename = void>
		struct is_stream_writable : std::false_type {};

		template<typename S, typename T>
		struct is_stream_writable<S, T,
			std::void_t<  decltype(std::declval<S&>() << std::declval<T>())  > >
			: std::true_type {};

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
				_out << _any.type().name();
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

	class ExtAny
	{
	public:
		std::any any;

		ExtAny() = default;
		ExtAny(const ExtAny& _oth) = default;
		ExtAny(ExtAny&& _oth) = default;

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

		friend std::ostream& operator<<(std::ostream& _out, const ExtAny& _any);
		// works only when already initialized with a type
		friend std::istream& operator>>(std::istream& _out, ExtAny& _any);

	private:
		using PrintFn = void(std::ostream&, const std::any&);
		using ReadFn = void(std::istream&, std::any&);

		PrintFn* m_print = nullptr;
		ReadFn* m_read = nullptr;
	};

	class HyperParams
	{
	public:

		ExtAny& operator[](const std::string& key) { return data[key]; }

		template<typename T>
		T get(const std::string& _key, const T& _defaultValue) const
		{
			const auto it = data.find(_key);
			if (std::optional<T> opt; it != data.end() && (opt = cast<T>(it->second.any)).has_value())
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
				return cast<T>(it->second.any);
			else return {};
		}

		friend std::ostream& operator<<(std::ostream& _out, const HyperParams& _params);
		friend std::istream& operator>>(std::istream& _in, HyperParams& _params);

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

		std::unordered_map<std::string, ExtAny> data;
	};

	using HyperParamGrid = std::vector<std::pair<std::string, std::vector<ExtAny>>>;
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

}

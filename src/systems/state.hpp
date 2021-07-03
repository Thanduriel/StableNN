#pragma once

#include <array>
#include <iostream>

namespace systems {

	template<typename T, size_t N, size_t M>
	class Matrix : public std::array<T, N * M>
	{
	public:
		constexpr Matrix<T, N, M>& operator+=(const Matrix<T, N, M>& oth)
		{
			for (size_t i = 0; i < N * M; ++i)
				(*this)[i] += oth[i];
			return *this;
		}

	//	using std::array<T, N * M>::array;
	//	constexpr Matrix(std::initializer_list<T> _init) : std::array<T, N * M>{_init} {}
	};

	// using std::array as a simple mathematical vector
	template<typename T, std::size_t N>
	using Vec = Matrix<T,N,1>;

	template<typename T, std::size_t N, std::size_t M>
	constexpr Matrix<T, N, M> operator+(const Matrix<T, N, M>& a, const Matrix<T, N, M>& b)
	{
		Matrix<T, N, M> result{}; // initialization should not be necessary, but is need for the constexpr case
		for (size_t i = 0; i < N*M; ++i)
			result[i] = a[i] + b[i];
		return result;
	}

	template<typename T, std::size_t N, std::size_t M, size_t K>
	constexpr Matrix<T, N, K> operator*(const Matrix<T, N, M>& a, const Matrix<T, M, K>& b)
	{
		Matrix<T, N, K> result{};
		for (size_t n = 0; n < N; ++n)
		{
			const size_t ind = n * M;
			for (size_t k = 0; k < K; ++k)
			{
				T acc = a[ind] * b[k];
				for (size_t m = 1; m < M; ++m)
					acc += a[ind + m] * b[m * K + k];
				result[n * K + k] = acc;
			}
		}
		return result;
	}

	template<typename T, std::size_t N, std::size_t M>
	constexpr Matrix<T, N, M> operator*(const T s, const Matrix<T, N, M>& v)
	{
		Matrix<T, N, M> result{};
		for (size_t i = 0; i < N*M; ++i)
			result[i] = s * v[i];
		return result;
	}

	template<typename T, std::size_t N, std::size_t M>
	constexpr Matrix<T, N, M> operator*(const Matrix<T, N, M>& v, const T s)
	{
		return s * v;
	}

/*	template<typename T, std::size_t N, std::size_t M, typename Pred>
	constexpr Matrix<T, N, M>& apply(const Matrix<T, N, M>& m, Pred pred)
	{
		Matrix<T, N, M> result;
		for (size_t i = 0; i < N * M; ++i)
			result[i] = pred(m[i]);
		return result;
	}*/

	// Assuming the state only consists of floating point numbers
	template<typename System>
	constexpr std::size_t sizeOfState()
	{
		static_assert(sizeof(typename System::State) % sizeof(typename System::ValueT) == 0,
			"Inconsistent state size.");
		return sizeof(typename System::State) / sizeof(typename System::ValueT);
	}

	template<typename State>
	using ValueType = std::remove_cv_t<std::remove_reference_t<decltype(std::declval<State>()[0])>>;

	template<typename State>
	auto average(const State& _state)
	{
		using T = ValueType<State>;
		T sum = 0.0;
		for (auto v : _state)
			sum += v;
		return sum / _state.size();
	}

	template<typename T, typename State>
	State normalizeDistribution(const State& _state, T _mean) // , T _stdDev
	{
		const T mean = average(_state);

		const double shift = mean - _mean;
		State state = _state;
		for (auto& v : state)
			v -= shift;

		return state;
	}
}

template<typename T, std::size_t Size>
std::ostream& operator << (std::ostream& out, const std::array<T, Size>& s)
{
	for (const T& el : s)
		std::cout << el << " ";
	return out;
}
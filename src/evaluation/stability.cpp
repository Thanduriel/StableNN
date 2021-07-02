#include "stability.hpp"
#include "../constants.hpp"

namespace eval {

	double g_sideEffect = 0.0;

	class NullBuffer : public std::streambuf
	{
	public:
		int overflow(int c) { return c; }
	};

	NullBuffer nullBuffer;
	std::ostream g_nullStream(&nullBuffer);

	template<>
	void checkLayerStability<torch::nn::Linear>(const torch::nn::Linear& _layer)
	{
		const auto& [eigenvalues, _] = torch::eig(_layer->weight);
		std::cout << eigenvalues << "\n";
	}

	std::vector<std::complex<double>> computeEigs(const torch::Tensor& _tensor)
	{
		const auto& [eigenvalues, _] = torch::eig(_tensor);
		std::vector<std::complex<double>> vals;
		const int64_t n = eigenvalues.sizes()[0];
		vals.reserve(n);

		for (int64_t i = 0; i < n; ++i)
		{
			vals.emplace_back(eigenvalues.index({ i,0 }).item<double>(),
				eigenvalues.index({ i,1 }).item<double>());
		}
		return vals;
	}

	torch::Tensor toMatrix(const torch::nn::Conv1d& _conv, int64_t _size)
	{
		assert(c10::holds_alternative<torch::enumtype::kCircular>(_conv->options.padding_mode()));

		return toMatrix(_conv->weight, _size);
	}

	torch::Tensor toMatrix(const torch::Tensor& _kernel, int64_t _size)
	{
		using namespace torch::indexing;

		// layer weight requiring a gradient would otherwise spread to mat
		torch::NoGradGuard guard;

		const torch::Tensor filter = _kernel.squeeze();
		const int64_t filterSize = filter.sizes().front();
		const int64_t halfSize = filterSize / 2;

		assert(filterSize <= _size);

		auto options = _kernel.options();
		torch::Tensor mat = torch::zeros({ _size, _size }, options.requires_grad(false));

		for (int64_t i = 0; i < _size; ++i)
		{
			for (int64_t j = 0; j < filterSize; ++j)
			{
				int64_t idx = i + j - halfSize;
				if (idx < 0) idx += _size;
				if (idx >= _size) idx -= _size;
				mat.index_put_({ i,idx }, filter[j]);
			}
		}
		return mat;
	}

	template<typename T>
	T power(const T& a, int p)
	{
		if (p == 0) return 1.0;
		T x = a;

		for (int i = 0; i < p; ++i)
			x *= a;

		return x;
	}

	torch::Tensor eigs(const torch::nn::Conv1d& _conv, int64_t _size)
	{
		// layer weight requiring a gradient would otherwise spread to mat
		torch::NoGradGuard guard;

		const int64_t filterSize = _conv->options.kernel_size()->front();
		const int64_t halfSize = filterSize / 2;
		const torch::Tensor filter = _conv->weight.squeeze().roll({ halfSize }, {0});

		assert(c10::holds_alternative<torch::enumtype::kCircular>(_conv->options.padding_mode()));
		assert(filterSize <= _size);
		
		using namespace std::literals::complex_literals;
		const std::complex<double> w = std::exp(2.0 * PI / static_cast<double>(_size) * 1i);

		torch::Tensor vals = torch::zeros({_size, 2}, filter.options());
		for (int64_t i = 0; i < _size; ++i)
		{
			std::complex<double> lambda = 0.0;
			for (int64_t j = 0; j < filterSize; ++j)
			{
				lambda += filter[j].item<double>() * power(w, i * j);
			}
			vals[i] = torch::tensor({ lambda.real(), lambda.imag() }, filter.options());
		}
		return vals;
	}

	void checkEnergy(const torch::nn::Conv1d& _conv, int64_t _size)
	{
		checkEnergy(toMatrix(_conv, _size));

		const int64_t filterSize = _conv->options.kernel_size()->front();
		const int64_t halfSize = filterSize / 2;
		const torch::Tensor filter = _conv->weight.squeeze();
		bool symmetric = true;
		double sum = 0.0;
		for (int64_t j = 0; j < halfSize; ++j)
		{
			if ((filter[j] - filter[filterSize - j - 1]).item<double>() > std::numeric_limits<double>::epsilon() * 16.0)
			{
				symmetric = false;
				break;
			}
			sum += std::abs(filter[j].item<double>()) + std::abs(filter[filterSize - j - 1].item<double>());
		}

		std::cout << filter[halfSize].item<double>() + sum << "\n";
		std::cout << "symmetric: " << symmetric << "\n";
		//	std::cout << "Holds for any size: " << (sum < filter[halfSize].item<double>()) << std::endl;
	}

	std::vector<double> checkEnergy(const torch::Tensor& _netLinear, int64_t _stateSize)
	{
		assert(_netLinear.dim() == 2);
		assert(_netLinear.size(0) == _netLinear.size(1));

		using namespace torch::indexing;

		if (!_stateSize)
			_stateSize = _netLinear.size(0);
		const int64_t n = _netLinear.size(0);
	//	const torch::Tensor id = torch::eye(_stateSize, _netLinear.options());
		torch::Tensor id = torch::zeros_like(_netLinear);
		id.index_put_({ Slice(n - _stateSize), Slice(n - _stateSize) }, 
			torch::eye(_stateSize, _netLinear.options()));
		const auto mat = _netLinear.transpose(0, 1) * _netLinear - id;
		const auto eigs = computeEigs(mat);
		std::vector<double> eigsReal;
		for (auto eig : eigs)
			eigsReal.push_back(eig.real());
	//	std::cout << "eigenvalues of A^T A - I:\n" << eigs << "\n";

		double maxEig = std::numeric_limits<double>::min();
		for (int64_t i = 0; i < n; ++i)
		{
			const double eig = eigsReal[i];
			if ( eig > maxEig)
			{
				maxEig = eig;
			}
		}
		//const bool isPositiveDefinite = maxEig <= 0;
		//std::cout << "Energy does not increase: " << isPositiveDefinite << std::endl;
		std::cout << maxEig << "\n";

		return eigsReal;
	}
}
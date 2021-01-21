#include "stability.hpp"

namespace eval {

	template<>
	void checkLayerStability<torch::nn::Linear>(const torch::nn::Linear& _layer)
	{
		const auto& [eigenvalues, _] = torch::eig(_layer->weight);
		std::cout << eigenvalues << "\n";
	}

	torch::Tensor toMatrix(const torch::nn::Conv1d& _conv, int64_t _size)
	{
		using namespace torch::indexing;

		const int64_t filterSize = _conv->options.kernel_size()->front();
		const int64_t halfSize = filterSize / 2;
		const torch::Tensor filter = _conv->weight.squeeze();
		
		assert(c10::holds_alternative<torch::enumtype::kCircular>(_conv->options.padding_mode()));
		assert(filterSize <= _size);

		auto options = _conv->weight.options();
		torch::Tensor mat = torch::zeros({ _size, _size }, options.requires_grad(false));

		// layer weight requiring a gradient would otherwise spread to mat
		torch::NoGradGuard guard;
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

	void checkEnergy(const torch::nn::Conv1d& _conv, int64_t _size)
	{
		const auto mat = toMatrix(_conv, _size);
		const auto [eigs, _] = torch::eig(torch::eye(_size, mat.options()) - mat.transpose(0, 1) * mat);

		bool isPositiveDefinite = true;
		for (int64_t i = 0; i < _size; ++i)
		{
			if (eigs.index({ i,0 }).item<double>() < 0.0)
			{
				isPositiveDefinite = false;
				break;
			}
		}

		std::cout << "Energy does not increase: " << isPositiveDefinite << std::endl;
		
		const int64_t filterSize = _conv->options.kernel_size()->front();
		const int64_t halfSize = filterSize / 2;
		const torch::Tensor filter = _conv->weight.squeeze();
		bool symmetric = true;
		double sum = 0.0;
		for (int64_t j = 0; j < halfSize; ++j)
		{
		//	std::cout << (filter[j] - filter[filterSize - j - 1]).item<double>() << "\n";
			if ((filter[j] - filter[filterSize - j - 1]).item<double>() > std::numeric_limits<double>::epsilon() * 16.0)
			{
				symmetric = false;
			//	break;
			}
			sum += std::abs(filter[j].item<double>()) + std::abs(filter[filterSize - j - 1].item<double>());
		}

		std::cout << filter[halfSize].item<double>() + sum;
	//	std::cout << "Holds for any size: " << (sum < filter[halfSize].item<double>()) << std::endl;
	}
}
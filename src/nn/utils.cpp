#include "utils.hpp"
#include <sstream>
#include <fstream>

namespace nn {
	torch::Tensor shiftTimeSeries(const torch::Tensor& _old, const torch::Tensor& _newEntry, int _stateSize)
	{
		using namespace torch::indexing;
		const int64_t len = _old.size(1);
		torch::Tensor newInput = torch::zeros_like(_old);
		newInput.index_put_({ "...", Slice(0, len - _stateSize) },
			_old.index({ "...", Slice(_stateSize, len) }));
		newInput.index_put_({ "...", Slice(len - _stateSize, len) }, _newEntry);

		return newInput;
	}

	void exportTensor(const torch::Tensor& _tensor, const std::string& _fileName, bool _pgfPlotsFormat)
	{
		std::ofstream file(_fileName);
		file.precision(24);
		file << std::fixed;
		
		torch::Tensor tensor = _tensor.squeeze();
		if (tensor.ndimension() == 1)
			tensor.unsqueeze_(0);
		const auto size = tensor.sizes();
		for (int64_t i = 0; i < size[0]; ++i)
		{
			for (int64_t j = 0; j < size[1]; ++j)
			{
				if(_pgfPlotsFormat)
					file << j << " " << i << " " << tensor.index({ i,j }).item<double>() << "\n";
				else
					file << tensor.index({ i,j }).item<double>() << " ";
			}
			file << "\n";
		}
	//	file << content;
	}
}
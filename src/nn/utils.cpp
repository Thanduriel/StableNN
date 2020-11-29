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

	torch::Tensor lp_loss(const torch::Tensor& input, const torch::Tensor& target, c10::Scalar p)
	{
		return (input - target).norm(p, { 1 }).mean();
	}

	void exportTensor(const torch::Tensor& _tensor, const std::string& _fileName)
	{
		std::stringstream ss;
		at::print(ss, _tensor, 0xffffff);

		std::string content = ss.str();
		const size_t end = content.find_first_of('[');
		content = content.substr(0, end);

		std::ofstream file(_fileName);
		file << content;
	}
}
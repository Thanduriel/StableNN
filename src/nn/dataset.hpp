#pragma once

#include <torch/torch.h>

namespace nn {

	class Dataset : public torch::data::Dataset<Dataset>
	{
	public:
		Dataset(torch::Tensor _inputs, torch::Tensor _outputs);

		torch::data::Example<> get(size_t index) override;
		c10::optional<size_t> size() const override;

	private:
		torch::Tensor m_inputs;
		torch::Tensor m_outputs;
	};
}
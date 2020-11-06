#include "dataset.hpp"

namespace nn {

	Dataset::Dataset(torch::Tensor _inputs, torch::Tensor _outputs)
		: m_inputs(std::move(_inputs)), m_outputs(std::move(_outputs))
	{
	}

	torch::data::Example<> Dataset::get(size_t index)
	{
		return { m_inputs[index], m_outputs[index] };
	}

	c10::optional<size_t> Dataset::size() const
	{
		return m_inputs.sizes()[0];
	}
}
#include "hamiltonian.hpp"

namespace nn {

	using namespace torch;

	HamiltonianImpl::HamiltonianImpl(const HamiltonianOptions& _options)
		: options(_options)
	{
		reset();
	}

	void HamiltonianImpl::reset()
	{
		layers.clear();
		layers.reserve(options.num_layers());

		timeStep = options.total_time() / options.num_layers();
		for (int64_t i = 0; i < options.num_layers(); ++i)
		{
			layers.emplace_back(torch::nn::LinearOptions(options.input_size(), options.input_size()).bias(options.bias()));
			register_module("hidden" + std::to_string(i), layers.back());
		}
	}

	torch::Tensor HamiltonianImpl::forward(const Tensor& _input)
	{
		auto& activation = options.activation();

		Tensor x0 = _input;
		Tensor x1 = x0 + 0.5 * timeStep * timeStep * activation(layers[0](x0));
		bool flipped = false;

		for (auto it = layers.begin() + 1; it != layers.end(); ++it)
		{
			auto& layer = *it;
			Tensor& y0 = flipped ? x1 : x0;
			Tensor& y1 = flipped ? x0 : x1;
			y0 = 2.0 * y1 - y0 + timeStep * timeStep * activation(layer(y1));
			flipped = !flipped;
		}

		return flipped ? x0 : x1;

	}


	// ****************************************************************** //
	HamiltonianCellImpl::HamiltonianCellImpl(int64_t _stateSize, int64_t _augmentSize, bool _bias)
		: size(_stateSize), augmentSize(_augmentSize), useBias(_bias)
	{
		reset();
	}

	void HamiltonianCellImpl::reset()
	{
		weight = register_parameter("weight",
			torch::empty({ size, augmentSize }));
		if (useBias) {
			biasY = register_parameter("biasY", torch::empty(size));
			biasZ = register_parameter("biasZ", torch::empty(augmentSize));
		}
		else {
			biasY = register_parameter("biasY", {}, false);
			biasZ = register_parameter("biasZ", {}, false);
		}

		reset_parameters();
	}

	void HamiltonianCellImpl::reset_parameters()
	{
		torch::nn::init::kaiming_uniform_(weight, std::sqrt(5));
		if (biasY.defined())
		{
			const auto& [fan_in, fan_out] =
				torch::nn::init::_calculate_fan_in_and_fan_out(weight);
			const auto bound = 1 / std::sqrt(fan_in);
			torch::nn::init::uniform_(biasY, -bound, bound);
			torch::nn::init::uniform_(biasZ, -bound, bound);
		}
	}

	void HamiltonianCellImpl::pretty_print(std::ostream& stream) const {
		stream << std::boolalpha
			<< "nn::Hamiltonian(size=" << size
			<< ", augmentSize=" << augmentSize
			<< ", bias=" << useBias << ")";
	}

	Tensor HamiltonianCellImpl::forwardY(const Tensor& input)
	{
		return torch::nn::functional::linear(input, weight.t(), biasZ);
	}

	Tensor HamiltonianCellImpl::forwardZ(const Tensor& input)
	{
		return torch::nn::functional::linear(input, weight, biasY);
	}

	HamiltonianAugmentedImpl::HamiltonianAugmentedImpl(const HamiltonianOptions& _options)
		: options(_options)
	{
		reset();
	}

	void HamiltonianAugmentedImpl::reset()
	{
		layers.clear();
		layers.reserve(options.num_layers());

		timeStep = options.total_time() / options.num_layers();
		
		for (int64_t i = 0; i < options.num_layers(); ++i)
		{
			layers.emplace_back(options.input_size(), options.augment_size(), options.bias());
			register_module("layer" + std::to_string(i), layers.back());
		}
	}

	torch::Tensor HamiltonianAugmentedImpl::forward(const Tensor& _input)
	{
		auto size = _input.sizes().vec();
		size.back() = options.augment_size();
		Tensor z = torch::zeros(size, c10::TensorOptions(c10::kDouble)); // z_{j-1/2}
		Tensor y = _input; // y_j

		auto& activation = options.activation();
		for (auto& layer : layers)
		{
			z = z - timeStep * activation(layer->forwardY(y));
			y = y + timeStep * activation(layer->forwardZ(z));
		}

		return y;
	}

	// ****************************************************************** //
	HamiltonianInterleafedImpl::HamiltonianInterleafedImpl(const HamiltonianOptions& _options)
		: options(_options)
	{
		reset();
	}

	void HamiltonianInterleafedImpl::reset()
	{
		layers.clear();
		layers.reserve(options.num_layers());

		timeStep = options.total_time() / options.num_layers();

		const int64_t halfSize = options.input_size() / 2;
		for (int64_t i = 0; i < options.num_layers(); ++i)
		{
			layers.emplace_back(halfSize, halfSize, options.bias());
			register_module("layer" + std::to_string(i), layers.back());
		}
	}

	torch::Tensor HamiltonianInterleafedImpl::forward(const Tensor& _input)
	{
		using namespace torch::indexing;
		auto size = _input.sizes().vec();
		size.back() = options.augment_size();
		const int64_t halfSize = options.input_size() / 2;
		Tensor y = _input.dim() == 2 ? _input.index({ "...", Slice(0, halfSize) }) : _input.index({ Slice(0, halfSize) });
		Tensor z = _input.dim() == 2 ? _input.index({ "...", Slice(halfSize) }) : _input.index({ Slice(halfSize) });
	//	Tensor z = _input.index({"...", Slice(1, c10::nullopt, 2)}); // z_{j-1/2}
	//	Tensor y = _input.index({ "...", Slice(0, c10::nullopt, 2) });; // y_j

		auto& activation = options.activation();
		for (auto& layer : layers)
		{
			z = z - timeStep * activation(layer->forwardY(y));
			y = y + timeStep * activation(layer->forwardZ(z));
		}

		Tensor combined = torch::cat({ y,z }, _input.dim()-1);
	//	Tensor combined = torch::zeros_like(_input);
	//	combined.index_put_({ "...", Slice(1, c10::nullopt, 2) }, z);
	//	combined.index_put_({ "...", Slice(0, c10::nullopt, 2) }, y);
		return combined;
	}
}
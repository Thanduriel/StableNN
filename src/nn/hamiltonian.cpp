#include "hamiltonian.hpp"

namespace nn {

	using namespace torch;

	HamiltonianNet::HamiltonianNet(int64_t _inputs, int64_t _hiddenLayers, double _totalTime, bool _useBias, ActivationFn _activation)
		: timeStep(_totalTime / _hiddenLayers),
		activation(std::move(_activation))
	{
		for (int64_t i = 0; i < _hiddenLayers; ++i)
		{
			layers.emplace_back(torch::nn::LinearOptions(_inputs, _inputs).bias(_useBias));
			register_module("hidden" + std::to_string(i), layers.back());
		}
	}

	torch::Tensor HamiltonianNet::forward(const Tensor& _input)
	{
		Tensor y0 = _input;
		Tensor y1 = y0 + 0.5 * timeStep * timeStep * activation(layers[0](_input));
		for (auto it = layers.begin()+1; it != layers.end(); ++it)
		{
			auto& layer = *it;
			Tensor yTemp = y1.clone();
			y1 = 2.0 * y1 - y0 + timeStep * timeStep * activation(layer(y1));
		//	y1 = 2.0 * y1 - y0 + timeStep * timeStep * activation(layer(y1));
			y0 = yTemp;
		}

		return y1;
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
			auto& [fan_in, fan_out] =
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
		timeStep = options.total_time() / options.num_layers();
		layers.reserve(options.num_layers());
		for (int64_t i = 0; i < options.num_layers(); ++i)
		{
			layers.emplace_back(options.input_size(), options.augment_size(), options.bias());
			register_module("layer" + std::to_string(i), layers.back());
		}
	}

	template<typename T>
	T loadValue(torch::serialize::InputArchive& archive, const std::string& name)
	{
		c10::IValue value;
		archive.read(name, value);

		return value.to<T>();
	}

	
/*	HamiltonianAugmentedNet::HamiltonianAugmentedNet(torch::serialize::InputArchive& archive)
		: HamiltonianAugmentedNet(
			loadValue<int64_t>(archive, "inputs"),
			loadValue<int64_t>(archive, "hiddenLayers"),
			loadValue<double>(archive, "totalTime"),
			false,
			torch::tanh,
			loadValue<int64_t>(archive, "augmentSize"))
	{
	}*/
	
	/*void HamiltonianAugmentedImpl::save(torch::serialize::OutputArchive& archive) const
	{
		archive.write("inputs", c10::IValue(options.input_size));
		archive.write("hiddenLayers", c10::IValue(static_cast<int64_t>(layers.size())));
		archive.write("totalTime", c10::IValue(timeStep * layers.size()));
		archive.write("augmentSize", c10::IValue(augmentSize));

		torch::nn::Module::save(archive);
	}*/

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
}
#include "mlp.hpp"

namespace nn {

	MultiLayerPerceptron::MultiLayerPerceptron(int64_t _inputs, int64_t _outputs)
		: linear1(_inputs, 32),
		linear2(32, 32),
		linear3(32, _outputs)
	{
		register_module("linear1", linear1);
		register_module("linear2", linear2);
		register_module("linear3", linear3);
	}

	torch::Tensor MultiLayerPerceptron::forward(torch::Tensor x)
	{
		x = torch::sigmoid(linear1(x));
		x = torch::relu(linear2(x));
		x = linear3(x);
		return x;
	}


}
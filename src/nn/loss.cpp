#include "loss.hpp"


namespace nn {
	torch::Tensor lpLoss(const torch::Tensor& input, const torch::Tensor& target, c10::Scalar p)
	{
		assert(input.squeeze().dim() <= 2); // currently no support for multidimensional data
		torch::Tensor dif = input - target;
		return (input - target).norm(p, dif.dim() - 1).mean() * 100;
	}

	torch::Tensor energyLoss(const torch::Tensor& netInput, const torch::Tensor& netOutput)
	{
		torch::Tensor energyDif = (netOutput.square().sum(netInput.dim()-1) 
			- netInput.square().sum(netOutput.dim()-1));
		return energyDif.clamp_min(0.0).mean();
	}

	LossFn makeLossFunction(const HyperParams& _params, bool _train)
	{
		const int p = _params.get<int>("loss_p", 2);
		const int e = _params.get<double>("loss_energy", 0.0);

		if (e == 0.0 || !_train)
		{
			return [p](const torch::Tensor& self, const torch::Tensor& target, const torch::Tensor& data)
			{
				return nn::lpLoss(self, target, p);
			};
		}
		else
		{
			return [p, e](const torch::Tensor& self, const torch::Tensor& target, const torch::Tensor& data)
			{
				return nn::lpLoss(self, target, p) + e * energyLoss(data, self);
			};
		}
	}

}
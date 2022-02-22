#pragma once
#include "Constants.h"
class ConnectedLayer
{
public:
	ConnectedLayer(std::vector<float> inputs, std::vector<float> weights, std::vector<float>biases, int nodes, ACTIVATION activation);

	void CalculateOutputs();
	void SetInputs(std::vector<float> inputs);
	std::vector<float> GetOutputs();

private:
	Layer layer;
};


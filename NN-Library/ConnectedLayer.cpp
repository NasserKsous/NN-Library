#include "ConnectedLayer.h"

ConnectedLayer::ConnectedLayer()
{
}

ConnectedLayer::ConnectedLayer(std::vector<float> in, std::vector<float> wei, std::vector<float> bi, int no, ACTIVATION act)
{
	activation = act;
	inputs = in;
	biases = bi;
	weights = wei;
	nodes = no;

}

void ConnectedLayer::CalculateOutputs()
{
	int numberOfWeights = weights.size();
	int numOfWeightsPerNode = numberOfWeights / nodes;
	for (int i = 0; i < nodes; ++i)
	{
		float tempOutput = 0;
		for (int j = 0; j < numOfWeightsPerNode; ++j)
		{
			tempOutput += inputs[j] * weights[j + (i * numOfWeightsPerNode)];
		}
		tempOutput += biases[i];
		ActivationFunction activeFunc(activation);
		tempOutput = activeFunc.Activate(tempOutput);
		outputs.push_back(tempOutput);
	}
}

void ConnectedLayer::SetInputs(std::vector<float> in)
{
	inputs = in;
}

std::vector<float> ConnectedLayer::GetOutputs()
{
	return outputs;
}

float ConnectedLayer::Activate(float input)
{
	return 0.0f;
}

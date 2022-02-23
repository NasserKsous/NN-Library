#include "ConnectedLayer.h"

ConnectedLayer::ConnectedLayer()
{
}

ConnectedLayer::ConnectedLayer(std::vector<float> inputs, std::vector<float> weights, std::vector<float> biases, int nodes, ACTIVATION activation)
{
	layer.activation = activation;
	layer.inputs = inputs;
	layer.biases = biases;
	layer.weights = weights;
	layer.nodes = nodes;
}

void ConnectedLayer::CalculateOutputs()
{
	int numberOfWeights = layer.weights.size();
	int numOfWeightsPerNode = numberOfWeights / layer.nodes;
	for (int i = 0; i < layer.nodes; ++i)
	{
		float tempOutput = 0;
		for (int j = 0; j < numOfWeightsPerNode; ++j)
		{
			tempOutput += layer.inputs[j] * layer.weights[j + (i * numOfWeightsPerNode)];
		}
		tempOutput += layer.biases[i];
		ActivationFunction activeFunc(layer.activation);
		tempOutput = activeFunc.Activate(tempOutput);
		layer.outputs.push_back(tempOutput);
	}
}

void ConnectedLayer::SetInputs(std::vector<float> inputs)
{
	layer.inputs = inputs;
}

std::vector<float> ConnectedLayer::GetOutputs()
{
	return layer.outputs;
}

Layer ConnectedLayer::GetLayer()
{
	return layer;
}

float ConnectedLayer::Activate(float input)
{
	return 0.0f;
}

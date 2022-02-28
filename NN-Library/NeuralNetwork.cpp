#include "NeuralNetwork.h"

#include <assert.h>

void NeuralNetwork::AddLayer(Layer layerToAdd)
{
	Network.push_back(layerToAdd);
	numberOfLayers = Network.size();
}

void NeuralNetwork::SetInputs(std::vector<float> inputs)
{
	assert(numberOfLayers <= 0 && "There are no layers to set inputs for.");

	Network[0].inputs = inputs;
}

void NeuralNetwork::CalculateOutputs()
{
	std::vector<float> inputs = Network[0].inputs;
	for (int i = 0; i < numberOfLayers; ++i)
	{
		Network[i].CalculateOutputs();
		inputs = Network[i].outputs;
	}
	outputs = inputs;
}

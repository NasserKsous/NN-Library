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
	for (int i = 0; i < numberOfLayers; ++i)
	{
		Network[0].
	}
}

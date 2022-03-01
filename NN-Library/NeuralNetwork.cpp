#include "NeuralNetwork.h"
#include <assert.h>

NeuralNetwork::NeuralNetwork()
{
	numberOfLayers = 0;
}

NeuralNetwork::~NeuralNetwork()
{
	Network.clear();
}

void NeuralNetwork::AddLayer(Layer* layerToAdd)
{
	Network.push_back(layerToAdd);
	numberOfLayers = Network.size();
}

void NeuralNetwork::SetInputs(std::vector<float> inputs)
{
	assert(numberOfLayers > 0 && "There are no layers to set inputs for.");

	Network[0]->inputs = inputs;
}

void NeuralNetwork::CalculateOutputs()
{
	assert(numberOfLayers > 0 && "There are no layers to calculate outputs for.");

	std::vector<float> inputs = Network[0]->inputs;
	for (int i = 0; i < numberOfLayers; ++i)
	{
		Network[i]->inputs = inputs;
		Network[i]->CalculateOutputs();
		inputs = Network[i]->outputs;
	}
	outputs = inputs;
}

int NeuralNetwork::GetNumberOfLayers()
{
	return numberOfLayers;
}

std::vector<float> NeuralNetwork::GetOutputs()
{
	return outputs;
}

std::vector<Layer*> NeuralNetwork::GetNetwork()
{
	return Network;
}

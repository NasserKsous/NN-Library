#include "NeuralNetwork.h"
#include <assert.h>

//Reference for back-prop: https://blog.demofox.org/2017/03/09/how-to-train-neural-networks-with-backpropagation/


NeuralNetwork::NeuralNetwork()
{
	numberOfLayers = 0;
	setsOfInputs = 1;
}

NeuralNetwork::NeuralNetwork(int sets)
{
	numberOfLayers = 0;
	setsOfInputs = sets;
}

NeuralNetwork::~NeuralNetwork()
{
	Network.clear();
}

void NeuralNetwork::AddLayer(Layer* layerToAdd)
{
	Network.push_back(layerToAdd);
	numberOfLayers = Network.size();

	int numOfWeights = layerToAdd->weights.size();

	for (int i = 0; i < numOfWeights; ++i)
		weights.push_back(layerToAdd->weights[i]);
	
	int numOfBiases = layerToAdd->biases.size();

	for (int i = 0; i < numOfBiases; ++i)
		biases.push_back(layerToAdd->biases[i]);
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

void NeuralNetwork::BackPropagate(std::vector<float> expectedOutputs)
{
	cost = 0.0f;
	biasesCosts.clear();
	weightsCosts.clear();

	int numberOfOutputs = outputs.size();
	for (int i = 0; i < numberOfOutputs; ++i)
	{
		cost += (expectedOutputs[i] - outputs[i]) * (expectedOutputs[i] - outputs[i]);
	}
	cost /= numberOfOutputs;

	Network[numberOfLayers - 1]->BackPropagate(expectedOutputs);

	for (int i = numberOfLayers - 2; i >= 0; --i)
	{
		Network[i]->BackPropagate(Network[i+1]->GetBiasCosts(), Network[i + 1]->weights);
	}
	
	for (int i = numberOfLayers - 1; i >= 0; --i)
	{
		Network[i]->UpdateWeightsAndBiases();
	}
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

float NeuralNetwork::GetCost()
{
	return cost;
}

void NeuralNetwork::ResetValues()
{
	for (int layerIndex = 0; layerIndex < numberOfLayers; ++layerIndex)
	{
		Network[layerIndex]->ResetValues();
	}
}

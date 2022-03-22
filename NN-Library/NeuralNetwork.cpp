#include "NeuralNetwork.h"
#include <assert.h>

//Reference for back-prop: https://blog.demofox.org/2017/03/09/how-to-train-neural-networks-with-backpropagation/


NeuralNetwork::NeuralNetwork()
{
	numberOfLayers = 0;
	trainingSets = 1;
}

NeuralNetwork::NeuralNetwork(int sets)
{
	numberOfLayers = 0;
	trainingSets = sets;
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

std::vector<float> NeuralNetwork::CalculateOutputs()
{
	assert(numberOfLayers > 0 && "There are no layers to calculate outputs for.");

	std::vector<float> inputs = Network[0]->inputs;
	for (int i = 0; i < numberOfLayers; ++i)
	{
		Network[i]->inputs = inputs;
		Network[i]->CalculateOutputs();
		inputs = Network[i]->outputs;
	}
	return inputs;
}

void NeuralNetwork::BackPropagate(std::vector<float> expectedOutputs)
{
	/*cost = 0.0f;

	int numberOfOutputs = outputs.size();
	for (int i = 0; i < numberOfOutputs; ++i)
	{
		cost += (expectedOutputs[i] - outputs[i]) * (expectedOutputs[i] - outputs[i]);
	}
	cost /= numberOfOutputs;

	*/

	std::vector<float> tempBiasesCost;
	std::vector<float> tempWeightsCost;
	
	Network[numberOfLayers - 1]->ResetValues();
	Network[numberOfLayers - 1]->BackPropagate(expectedOutputs);

	for (int i = numberOfLayers - 2; i >= 0; --i)
	{
		Network[i]->ResetValues();
		Network[i]->BackPropagate(Network[i+1]->GetBiasCosts(), Network[i + 1]->weights);
	}

	tempBiasesCost = Network[0]->GetBiasCosts();
	tempWeightsCost = Network[0]->GetWeightCosts();

	for (int i = 1; i < numberOfLayers; ++i)
	{
		std::vector<float> temp = Network[i]->GetBiasCosts();
		std::vector<float> temp2 = Network[i]->GetWeightCosts();
		tempBiasesCost.insert(std::end(tempBiasesCost), std::begin(temp), std::end(temp));
		tempWeightsCost.insert(std::end(tempWeightsCost), std::begin(temp2), std::end(temp2));
	}
	
	if (biasesCosts.size() != 0)
	{
		for (int biasesCostsIndex = 0; biasesCostsIndex < biasesCosts.size(); ++biasesCostsIndex)
		{
			biasesCosts[biasesCostsIndex] += tempBiasesCost[biasesCostsIndex];
		}

		for (int weightsCostsIndex = 0; weightsCostsIndex < weightsCosts.size(); ++weightsCostsIndex)
		{
			weightsCosts[weightsCostsIndex] += tempWeightsCost[weightsCostsIndex];
		}
	}
	else
	{
		biasesCosts = tempBiasesCost;
		weightsCosts = tempWeightsCost;
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

void NeuralNetwork::TrainNetwork(std::vector<float> inputs, std::vector<float> expectedOutputs)
{
	int numOfInputs = inputs.size();
	int inputsPerSet = numOfInputs / trainingSets;

	int numOfOutputs = expectedOutputs.size();
	int outputsPerSet = numOfOutputs / trainingSets;
	int outputsIndex = 0;

	std::vector<float> tempOutputs;

	for (int setIndex = 0; setIndex < numOfInputs; setIndex+=inputsPerSet)
	{
		std::vector<float> tempInputs(inputs.cbegin() + setIndex, inputs.cbegin() + setIndex + inputsPerSet);
		std::vector<float> tempExpectedOutputs(expectedOutputs.cbegin() + outputsIndex, expectedOutputs.cbegin() + outputsIndex + outputsPerSet);
		
		SetInputs(tempInputs);
		tempOutputs = CalculateOutputs();
		BackPropagate(tempExpectedOutputs);

		if (setIndex != 0)
		{
			outputs.insert(std::end(outputs), std::begin(tempOutputs), std::end(tempOutputs));
		}
		else
		{
			outputs = tempOutputs;
		}
		outputsIndex += outputsPerSet;
	}

	cost = 0.0f;

	int numberOfOutputs = outputs.size();
	for (int i = 0; i < numberOfOutputs; ++i)
	{
		cost += (expectedOutputs[i] - outputs[i]) * (expectedOutputs[i] - outputs[i]);
	}
	cost /= numberOfOutputs;

	for (int biasesCostsIndex = 0; biasesCostsIndex < biasesCosts.size(); ++biasesCostsIndex)
	{
		biasesCosts[biasesCostsIndex] /= trainingSets;
	}

	for (int weightsCostsIndex = 0; weightsCostsIndex < weightsCosts.size(); ++weightsCostsIndex)
	{
		weightsCosts[weightsCostsIndex] /= trainingSets;
	}

	UpdateWeightsAndBiases();
}

void NeuralNetwork::UpdateWeightsAndBiases()
{
	int weightCount = 0;
	int biasCount = 0;

	for (int layerIndex = 0; layerIndex < numberOfLayers; ++layerIndex)
	{
		int numOfWeights = Network[layerIndex]->weights.size();
		int numOfBiases = Network[layerIndex]->biases.size();

		std::vector<float> tempWeights(weightsCosts.cbegin() + weightCount, weightsCosts.cbegin() + weightCount + numOfWeights);
		std::vector<float> tempBiases(biasesCosts.cbegin() + biasCount, biasesCosts.cbegin() + biasCount + numOfBiases);

		Network[layerIndex]->UpdateWeightsAndBiases(tempWeights, tempBiases);

		weightCount += numOfWeights;
		biasCount += numOfBiases;
	}
}

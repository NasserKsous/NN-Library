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
	// Add layer to the vector array and update the number of layers.
	Network.push_back(layerToAdd);
	numberOfLayers = Network.size();
}

void NeuralNetwork::SetInputs(std::vector<float> inputs)
{
	// Check that there is at least one layer.
	assert(numberOfLayers > 0 && "There are no layers to set inputs for.");

	// Set the inputs of the first layer.
	Network[0]->inputs = inputs;
}

std::vector<float> NeuralNetwork::CalculateOutputs()
{
	// Check that there is at least one layer.
	assert(numberOfLayers > 0 && "There are no layers to calculate outputs for.");

	// Initialize the current outputs.
	std::vector<float> currentOutputs = Network[0]->inputs;

	// Calculate the outputs of each layer.
	for (int i = 0; i < numberOfLayers; ++i)
	{
		// Set the inputs of the layer as the outputs of the previous layer.
		Network[i]->SetInputs(currentOutputs);
		Network[i]->CalculateOutputs();
		currentOutputs = Network[i]->outputs;
	}

	// Return the outputs of the final layer.
	return currentOutputs;
}

void NeuralNetwork::BackPropagate(std::vector<float> expectedOutputs)
{
	// Initialize the temporary weight and bias costs for this training set.
	std::vector<float> tempBiasesCost;
	std::vector<float> tempWeightsCost;
	
	// Reset the bias and weight values of the last layer and then back propagate to caluclate the new bias and weight costs.
	Network[numberOfLayers - 1]->ResetValues();
	Network[numberOfLayers - 1]->BackPropagateLastLayer(expectedOutputs);

	// Repeat this for each layer going back to front in the network.
	for (int i = numberOfLayers - 2; i >= 0; --i)
	{
		Network[i]->ResetValues();
		Network[i]->BackPropagate(Network[i+1]->GetInputCosts());
	}

	// Set the temporary local weights and biases costs.
	tempBiasesCost = Network[0]->GetBiasCosts();
	tempWeightsCost = Network[0]->GetWeightCosts();

	// Add the other layer bias and weight costs to the back of the temporary arrays.
	for (int i = 1; i < numberOfLayers; ++i)
	{
		std::vector<float> layerBiasCosts = Network[i]->GetBiasCosts();
		std::vector<float> layerWeightCosts = Network[i]->GetWeightCosts();
		tempBiasesCost.insert(std::end(tempBiasesCost), std::begin(layerBiasCosts), std::end(layerBiasCosts));
		tempWeightsCost.insert(std::end(tempWeightsCost), std::begin(layerWeightCosts), std::end(layerWeightCosts));
	}
	
	// If the biases and weights costs exits for the network then add the temporary values to these, else set them as the new values.
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
	// Set the numder of inputs and the inputs per training set.
	int numOfInputs = inputs.size();
	int inputsPerSet = numOfInputs / trainingSets;

	// Set the number of expected outputs and the ouputs per training set.
	int numOfOutputs = expectedOutputs.size();
	int outputsPerSet = numOfOutputs / trainingSets;

	// Initialize the output index and the temporary local ouputs.
	int outputsIndex = 0;
	std::vector<float> tempOutputs;

	// For each training set.
	for (int setIndex = 0; setIndex < numOfInputs; setIndex+=inputsPerSet)
	{
		// Set the inputs and expected outputs for that set.
		std::vector<float> tempInputs(inputs.cbegin() + setIndex, inputs.cbegin() + setIndex + inputsPerSet);
		std::vector<float> tempExpectedOutputs(expectedOutputs.cbegin() + outputsIndex, expectedOutputs.cbegin() + outputsIndex + outputsPerSet);
		
		// Set the inputs for the network, calculate the outputs and backpropagate for this set.
		SetInputs(tempInputs);
		tempOutputs = CalculateOutputs();
		BackPropagate(tempExpectedOutputs);

		// Save the outputs of each training set.
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

	// Reset the cost.
	cost = 0.0f;

	// Calculate the cost using Mean Squared Average.
	int numberOfOutputs = outputs.size();
	for (int i = 0; i < numberOfOutputs; ++i)
	{
		cost += (expectedOutputs[i] - outputs[i]) * (expectedOutputs[i] - outputs[i]);
	}
	cost /= numberOfOutputs;

	// Divide the bias and weight costs by the number of training sets.
	for (int biasesCostsIndex = 0; biasesCostsIndex < biasesCosts.size(); ++biasesCostsIndex)
	{
		biasesCosts[biasesCostsIndex] /= trainingSets;
	}

	for (int weightsCostsIndex = 0; weightsCostsIndex < weightsCosts.size(); ++weightsCostsIndex)
	{
		weightsCosts[weightsCostsIndex] /= trainingSets;
	}

	// Update the weights and biases for the network.
	UpdateWeightsAndBiases();

	// Clear the network weight and biases costs.
	biasesCosts.clear();
	weightsCosts.clear();
}

void NeuralNetwork::UpdateWeightsAndBiases()
{
	int weightCount = 0;
	int biasCount = 0;

	// For each layer, find the corresponding weight and biases costs and use them to update the weights and biases.
	for (int layerIndex = 0; layerIndex < numberOfLayers; ++layerIndex)
	{
		int numOfWeights = 0;
		int numOfBiases = 0;

		numOfWeights = Network[layerIndex]->weights.size();
		numOfBiases = Network[layerIndex]->biases.size();
		


		std::vector<float> tempWeights(weightsCosts.cbegin() + weightCount, weightsCosts.cbegin() + weightCount + numOfWeights);
		std::vector<float> tempBiases(biasesCosts.cbegin() + biasCount, biasesCosts.cbegin() + biasCount + numOfBiases);

		Network[layerIndex]->UpdateWeightsAndBiases(tempWeights, tempBiases, 0.1f);

		weightCount += numOfWeights;
		biasCount += numOfBiases;
	}
}

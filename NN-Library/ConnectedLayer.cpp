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
	// Clear the outputs.
	outputs.clear();

	// Set the number of weights and the weights per node in the layer.
	int numberOfWeights = weights.size();
	int numOfWeightsPerNode = numberOfWeights / nodes;

	// For each output node, calculate the output for the node.
	for (int nodeIndex = 0; nodeIndex < nodes; ++nodeIndex)
	{
		// Set the node output to 0.
		float nodeOutput = 0;

		// Multiply the inputs with the corresponding weights for the node.
		for (int j = 0; j < numOfWeightsPerNode; ++j)
		{
			nodeOutput += inputs[j] * weights[j + (nodeIndex * numOfWeightsPerNode)];
		}

		// Add the bias for the node.
		nodeOutput += biases[nodeIndex];

		// Calculate the final output after passing it in the activation function.
		ActivationFunction activeFunc(activation);
		nodeOutput = activeFunc.Activate(nodeOutput);

		// Place this node output the list of outputs for the layer.
		outputs.push_back(nodeOutput);
	}
}

void ConnectedLayer::BackPropagate(std::vector<float> expectedOutputs)
{
	// Set the number of weights and the weights per node in the layer.
	int numberOfWeights = weights.size();
	int numOfWeightsPerNode = numberOfWeights / nodes;

	// For each node in the layer.
	for (int nodeIndex = 0; nodeIndex < nodes; ++nodeIndex)
	{
		//Calculate bias cost using (output-expectedOutput) * (output*(1 - expected output))
		float diff1 = outputs[nodeIndex] - expectedOutputs[nodeIndex];
		float diff2 = outputs[nodeIndex] * (1.0f - outputs[nodeIndex]);
		float biasCost = diff1 * diff2;

		//Add this to the bias costs array.
		biasesCosts.push_back(biasCost);

		//Calculate weight costs using bias cost * inputs.
		for (int i = 0; i < numOfWeightsPerNode; ++i)
		{
			float weightCost = biasCost * inputs[i];
			weightsCosts.push_back(weightCost);
		}
	}
}

void ConnectedLayer::BackPropagate(std::vector<float> previousBiasCosts, std::vector<float> previousWeights)
{
	// Set the number of weights and the weights per node in the layer.
	int numberOfWeights = weights.size();
	int numOfWeightsPerNode = numberOfWeights / nodes;

	// For each node in the layer.
	for (int nodeIndex = 0; nodeIndex < nodes; ++nodeIndex)
	{
		// Set the initial cost of the previous layer to 0. 
		float costOfPrevLayer = 0.0f;

		// Set the number of weights per node for the previous layer. 
		float numOfWeightsPerNode2 = previousWeights.size() / previousBiasCosts.size();

		// Calculate the cost of the previous layer using cost = previous bias cost * previous weights.
		for (int i = 0; i < previousBiasCosts.size(); ++i)
		{
			costOfPrevLayer += previousBiasCosts[i] * previousWeights[i * numOfWeightsPerNode2 + nodeIndex];
		}

		// Calculate the bias cost using cost of previous layer * outputs * (1 - outputs).
		float outputsCost = outputs[nodeIndex] * (1.0f - outputs[nodeIndex]);
		float biasCost = costOfPrevLayer * outputsCost;
		biasesCosts.push_back(biasCost);

		//Calculate weight costs using bias cost * inputs.
		for (int i = 0; i < numOfWeightsPerNode; ++i)
		{
			float weightCost = biasCost * inputs[i];
			weightsCosts.push_back(weightCost);
		}
	}
}

void ConnectedLayer::SetInputs(std::vector<float> in)
{
	inputs = in;
}

void ConnectedLayer::UpdateWeightsAndBiases(std::vector<float> expWeightsCosts, std::vector<float> expBiasesCosts)
{
	// Initialize the learning rate.
	float learningRate = 0.5f;

	// Set the number of weights and biases.
	int numOfWeights = expWeightsCosts.size();
	int numOfBiases = expBiasesCosts.size();

	// Adjust the weights using weights = weights - learning rate * weight costs.
	for (int weightIndex = 0; weightIndex < numOfWeights; ++weightIndex)
	{
		weights[weightIndex] -= learningRate * (expWeightsCosts[weightIndex]);
	}
	
	// Adjust the biases using biases = biases - learning rate * bias costs.
	for (int biasIndex = 0; biasIndex < numOfBiases; ++biasIndex)
	{
		biases[biasIndex] -= learningRate * (expBiasesCosts[biasIndex]);
	}
}

void ConnectedLayer::ResetValues()
{
	// Clear the bias and weight costs.
	biasesCosts.clear();
	weightsCosts.clear();
}

std::vector<float> ConnectedLayer::GetBiasCosts()
{
	return biasesCosts;
}

std::vector<float> ConnectedLayer::GetWeightCosts()
{
	return weightsCosts;
}

float ConnectedLayer::Activate(float input)
{
	return 0.0f;
}

#include "ConnectedLayer.h"

ConnectedLayer::ConnectedLayer()
{
	layerType = LAYER_TYPE::CONNECTED;
}

ConnectedLayer::ConnectedLayer(std::vector<float> in, std::vector<float> wei, std::vector<float> bi, int no, ACTIVATION act)
{
	activation = act;
	inputs = in;
	biases = bi;
	weights = wei;
	nodes = no;

	layerType = LAYER_TYPE::CONNECTED;
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
		for (int weightIndex = 0; weightIndex < numOfWeightsPerNode; ++weightIndex)
		{
			nodeOutput += inputs[weightIndex] * weights[weightIndex + (nodeIndex * numOfWeightsPerNode)];
		}

		// Add the bias for the node.
		nodeOutput += biases[nodeIndex];

		// Calculate the final output after passing it in the activation function.
		if (activation != ACTIVATION::SOFTMAX)
			nodeOutput = Activate(nodeOutput, activation);

		// Place this node output the list of outputs for the layer.
		outputs.push_back(nodeOutput);
	}

	if (activation == ACTIVATION::SOFTMAX)
	{
		outputs = ActivateArray(outputs);
	}
}

void ConnectedLayer::BackPropagateLastLayer(std::vector<float> expectedOutputs)
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
		for (int inputIndex = 0; inputIndex < numOfWeightsPerNode; ++inputIndex)
		{
			float weightCost = biasCost * inputs[inputIndex];
			weightsCosts.push_back(weightCost);
		}
	}

	CalculateInputCosts();
}

void ConnectedLayer::BackPropagate(std::vector<float> previousLayerCosts)
{
	// Set the number of weights and the weights per node in the layer.
	int numberOfWeights = weights.size();
	int numOfWeightsPerNode = numberOfWeights / nodes;

	// For each node in the layer.
	for (int nodeIndex = 0; nodeIndex < nodes; ++nodeIndex)
	{
		// Calculate the bias cost using cost of previous layer * outputs * (1 - outputs).
		float outputsCost = outputs[nodeIndex] * (1.0f - outputs[nodeIndex]);
		float biasCost = previousLayerCosts[nodeIndex] * outputsCost;
		biasesCosts.push_back(biasCost);

		//Calculate weight costs using bias cost * inputs.
		for (int i = 0; i < numOfWeightsPerNode; ++i)
		{
			float weightCost = biasCost * inputs[i];
			weightsCosts.push_back(weightCost);
		}
	}

	CalculateInputCosts();
}

void ConnectedLayer::CalculateInputCosts()
{
	int numInputs = inputs.size();

	for (int inputIndex = 0; inputIndex < numInputs; ++inputIndex)
	{
		// Set the initial cost of the previous layer to 0. 
		float costOfPrevLayer = 0.0f;

		// Set the number of weights per node for the previous layer. 
		int numOfWeightsPerNode2 = weights.size() / biasesCosts.size();

		// Calculate the cost of the previous layer using cost = previous bias cost * previous weights.
		for (int i = 0; i < biasesCosts.size(); ++i)
		{
			costOfPrevLayer += biasesCosts[i] * weights[i * (float)numOfWeightsPerNode2 + inputIndex];
		}
		inputsCosts.push_back(costOfPrevLayer);
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

void ConnectedLayer::UpdateWeightsAndBiases(std::vector<float> expWeightsCosts, std::vector<float> expBiasesCosts)
{
	// Initialize the learning rate.
	float learningRate = 0.1f;

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
	inputsCosts.clear();
}

std::vector<float> ConnectedLayer::GetBiasCosts()
{
	return biasesCosts;
}

std::vector<float> ConnectedLayer::GetWeightCosts()
{
	return weightsCosts;
}

std::vector<float> ConnectedLayer::GetInputCosts()
{
	return inputsCosts;
}

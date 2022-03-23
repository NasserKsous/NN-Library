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
	outputs.clear();

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

void ConnectedLayer::BackPropagate(std::vector<float> expectedOutputs)
{
	/*biasesCosts.clear();
	weightsCosts.clear();*/

	int numberOfWeights = weights.size();
	int numOfWeightsPerNode = numberOfWeights / nodes;

	for (int neuronIndex = 0; neuronIndex < nodes; ++neuronIndex)
	{
		//Update Bias 
		float temp1 = outputs[neuronIndex] - expectedOutputs[neuronIndex];
		float temp2 = outputs[neuronIndex] * (1.0f - outputs[neuronIndex]);
		float biasCost = temp1 * temp2;
		biasesCosts.push_back(biasCost);

		//Update Weights
		
		for (int i = 0; i < numOfWeightsPerNode; ++i)
		{
			float weightCost = biasCost * inputs[i];
			weightsCosts.push_back(weightCost);
		}
	}


}

void ConnectedLayer::BackPropagate(std::vector<float> previousBiasCosts, std::vector<float> previousWeights)
{
	/*biasesCosts.clear();
	weightsCosts.clear();*/

	int numberOfWeights = weights.size();
	int numOfWeightsPerNode = numberOfWeights / nodes;

	for (int neuronIndex = 0; neuronIndex < nodes; ++neuronIndex)
	{
		//Update Bias 
		float costOfNextNeurons = 0.0f;

		float numOfWeightsPerNode2 = previousWeights.size() / previousBiasCosts.size();

		for (int i = 0; i < previousBiasCosts.size(); ++i)
		{
			costOfNextNeurons += previousBiasCosts[i] * previousWeights[i * numOfWeightsPerNode2 + neuronIndex];
		}

		float temp2 = outputs[neuronIndex] * (1.0f - outputs[neuronIndex]);
		float biasCost = costOfNextNeurons * temp2;
		biasesCosts.push_back(biasCost);

		//Update Weigths

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
	float learningRate = 0.5f;

	int numOfWeights = expWeightsCosts.size();
	int numOfBiases = expBiasesCosts.size();

	for (int weightIndex = 0; weightIndex < numOfWeights; ++weightIndex)
	{
		weights[weightIndex] -= learningRate * (expWeightsCosts[weightIndex]);
	}
	
	for (int biasIndex = 0; biasIndex < numOfBiases; ++biasIndex)
	{
		biases[biasIndex] -= learningRate * (expBiasesCosts[biasIndex]);
	}
}

void ConnectedLayer::ResetValues()
{
	//outputs.clear();
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

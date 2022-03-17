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
	biasesCosts.clear();
	weightsCosts.clear();
	

	int numberOfWeights = weights.size();
	int numOfWeightsPerNode = numberOfWeights / nodes;

	for (int neuronIndex = 0; neuronIndex < nodes; ++neuronIndex)
	{
		//Update Bias 
		float temp1 = outputs[neuronIndex] - expectedOutputs[neuronIndex];
		float temp2 = outputs[neuronIndex] * (1.0f - outputs[neuronIndex]);
		float biasCost = temp1 * temp2;
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

std::vector<float> ConnectedLayer::GetOutputs()
{
	return outputs;
}

void ConnectedLayer::UpdateWeightsAndBiases()
{
	float learningRate = 0.5f;
	int numberOfWeights = weights.size();
	int numOfWeightsPerNode = numberOfWeights / nodes;

	for (int neuronIndex = 0; neuronIndex < nodes; ++neuronIndex)
	{
		//Update Bias 
		biases[neuronIndex] -= learningRate * biasesCosts[neuronIndex];

		//Update Weigths

		for (int i = 0; i < numOfWeightsPerNode; ++i)
		{
			weights[i + (neuronIndex * numOfWeightsPerNode)] -= learningRate * weightsCosts[i + (neuronIndex * numOfWeightsPerNode)];
		}
	}
}

float ConnectedLayer::Activate(float input)
{
	return 0.0f;
}

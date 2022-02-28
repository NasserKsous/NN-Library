#pragma once
#include <iostream>
#include <vector>

enum class ACTIVATION
{
	LINEAR = 0,
	BINARY_STEP,
	SIGMOID,
	TANH,
	RELU,
	LEAKY_RELU,
	PARAMETRIC_RELU
};



struct Layer
{
	std::vector<float> inputs;
	std::vector<float> outputs;
	std::vector<float> weights;
	std::vector<float> biases;
	int nodes;
	ACTIVATION activation;

	virtual void CalculateOutputs()
	{

	}
};
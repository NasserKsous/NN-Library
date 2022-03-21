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



class Layer
{
public:
	std::vector<float> inputs;
	std::vector<float> outputs;
	std::vector<float> weights;
	std::vector<float> biases;
	int nodes;
	ACTIVATION activation;
	int setsOfInputs;

	virtual void CalculateOutputs() {};
	virtual void BackPropagate(std::vector<float> expectedOutputs) {};
	virtual void BackPropagate(std::vector<float> previousBiasCosts, std::vector<float> previousWeightCosts) {};
	virtual void UpdateWeightsAndBiases() {};
	virtual void ResetValues() {};
	virtual std::vector<float> GetBiasCosts() { return std::vector<float>(); };
	virtual std::vector<float> GetWeightCosts() { return std::vector<float>(); };
};
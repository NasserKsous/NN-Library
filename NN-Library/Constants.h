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
	LEAKY_RELU
};

enum class LAYER_TYPE
{
	CONNECTED = 0,
	CONVOLUTIONAL,
	POOLING
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
	LAYER_TYPE layerType;
	int setsOfInputs;

	virtual void SetInputs(std::vector<float> in) {};
	virtual void CalculateOutputs() {};
	virtual void BackPropagate(std::vector<float> expectedOutputs) {};
	virtual void BackPropagate(std::vector<float> previousBiasCosts, std::vector<float> previousWeightCosts) {};
	virtual void UpdateWeightsAndBiases(std::vector<float> expWeightsCosts, std::vector<float> expBiasesCosts) {};
	virtual void ResetValues() {};
	virtual std::vector<float> GetOutputs() { return std::vector<float>(); };
	virtual std::vector<float> GetBiasCosts() { return std::vector<float>(); };
	virtual std::vector<float> GetWeightCosts() { return std::vector<float>(); };
};
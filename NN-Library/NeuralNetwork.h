#pragma once
#include "Constants.h"

class NeuralNetwork
{
public:
	NeuralNetwork();
	NeuralNetwork(int sets);
	~NeuralNetwork();

	void AddLayer(Layer* layerToAdd);
	void SetInputs(std::vector<float> inputs);
	void CalculateOutputs();
	void BackPropagate(std::vector<float> expectedOutputs);
	int GetNumberOfLayers();
	std::vector<float> GetOutputs();
	std::vector<Layer*> GetNetwork();
	float GetCost();

	void ResetValues();

private:
	std::vector<Layer*> Network;
	int numberOfLayers;
	std::vector<float> outputs;

	float cost = 0.0f;
	int setsOfInputs;

	std::vector<float> weights;
	std::vector<float> biases;
	std::vector<float> weightsCosts;
	std::vector<float> biasesCosts;

};


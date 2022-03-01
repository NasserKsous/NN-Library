#pragma once
#include "Constants.h"

class NeuralNetwork
{
public:
	NeuralNetwork();
	~NeuralNetwork();

	void AddLayer(Layer* layerToAdd);
	void SetInputs(std::vector<float> inputs);
	void CalculateOutputs();
	int GetNumberOfLayers();
	std::vector<Layer*> GetNetwork();

private:
	std::vector<Layer*> Network;
	int numberOfLayers;
	std::vector<float> outputs;
};


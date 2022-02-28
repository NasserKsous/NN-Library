#pragma once
#include "Constants.h"

class NeuralNetwork
{
public:
	void AddLayer(Layer layerToAdd);

	void SetInputs(std::vector<float> inputs);

	void CalculateOutputs();

private:
	std::vector<Layer> Network;
	int numberOfLayers;
	std::vector<float> outputs;
};


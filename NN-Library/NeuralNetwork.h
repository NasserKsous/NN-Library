#pragma once
#include "Constants.h"

class NeuralNetwork
{
public:

	/* Default constructor that sets the number of training sets to 0. Used for networks that don't need to be trained. */
	NeuralNetwork();

	/* Constructor to set the number of training sets. Used for networks that need to be trained. */
	NeuralNetwork(int sets);

	/* Default deconstructor. */
	~NeuralNetwork();

	/* Adds a layer to the end of the network. */
	void AddLayer(Layer* layerToAdd);

	/* Sets the inputs for the first layer in the network. */
	void SetInputs(std::vector<float> in);

	/* Calculates the outputs of the network. */
	std::vector<float> CalculateOutputs();

	/* Calculates the cost values for all the weights and biases in the network.*/
	void BackPropagate(std::vector<float> expectedOutputs);

	/* Returns the number of layers in the network. */
	int GetNumberOfLayers();

	/* Returns the outputs of the network. */
	std::vector<float> GetOutputs();

	/* Returns the network as an array of layers. */
	std::vector<Layer*> GetNetwork();

	/* Returns the current cost of the network. */
	float GetCost();


	void ResetValues();

	/* Trains the network using the given inputs and expected outputs. */
	void TrainNetwork(std::vector<float> inputs, std::vector<float> expectedOutputs);

	/* Updates all the weights and biases with the corresponding cost values. */
	void UpdateWeightsAndBiases();

private:
	std::vector<Layer*> Network;
	int numberOfLayers;
	std::vector<float> outputs;

	float cost = 0.0f;
	int trainingSets;

	std::vector<float> weightsCosts;
	std::vector<float> biasesCosts;

};


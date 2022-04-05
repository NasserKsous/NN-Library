#pragma once
#include "Constants.h"
#include "ActivationFunctions.h"

class ConnectedLayer : public Layer
{
public:
	/* Default constructor for a connected layer. */
	ConnectedLayer();

	/* Constructor that takes in inputs, weights, biases, number of output nodes and activation function for the layer. */
	ConnectedLayer(std::vector<float> in, std::vector<float> wei, std::vector<float> bi, int no, ACTIVATION act);

	/* Calculates outputs for the connected layer. */
	void CalculateOutputs() override;

	/* Backpropagates the layer using the expected ouputs passed in. */
	void BackPropagate(std::vector<float> expectedOutputs) override;

	/* Backpropagates the layer using the previous layer's bias costs and weights. */
	void BackPropagate(std::vector<float> previousBiasCosts, std::vector<float> previousWeightCosts) override;

	/* Sets the inputs for the layer using the inputs passed in. */
	void SetInputs(std::vector<float> in) override;

	/* Returns the outputs of the layer. */
	std::vector<float> GetOutputs() override;

	/* Adjusts the weights and biases using the costs passed in. */
	void UpdateWeightsAndBiases(std::vector<float> expWeightsCosts, std::vector<float> expBiasesCosts) override;

	/* Resets the weight and biases costs. */
	void ResetValues() override;

	/* Returns the biases costs. */
	std::vector<float> GetBiasCosts() override;

	/* Returns the weights costs. */
	std::vector<float> GetWeightCosts() override;

private:

	std::vector<float> weightsCosts;
	std::vector<float> biasesCosts;
};


#pragma once
#include "Constants.h"
#include "ActivationFunctions.h"

class ConnectedLayer : public Layer
{
public:
	ConnectedLayer();
	ConnectedLayer(std::vector<float> in, std::vector<float> wei, std::vector<float> bi, int no, ACTIVATION act);

	void CalculateOutputs() override;
	void BackPropagate(std::vector<float> expectedOutputs) override;
	void BackPropagate(std::vector<float> previousBiasCosts, std::vector<float> previousWeightCosts) override;
	void SetInputs(std::vector<float> in);
	void UpdateWeightsAndBiases(std::vector<float> expWeightsCosts, std::vector<float> expBiasesCosts) override;
	void ResetValues() override;
	std::vector<float> GetBiasCosts() override;
	std::vector<float> GetWeightCosts() override;

private:
	float Activate(float input);

	std::vector<float> weightsCosts;
	std::vector<float> biasesCosts;
};


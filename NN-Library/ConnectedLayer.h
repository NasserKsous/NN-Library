#pragma once
#include "Constants.h"
#include "ActivationFunctions.h"

class ConnectedLayer : public Layer
{
public:
	ConnectedLayer();
	ConnectedLayer(std::vector<float> in, std::vector<float> wei, std::vector<float> bi, int no, ACTIVATION act);

	void CalculateOutputs() override;
	void SetInputs(std::vector<float> in);
	std::vector<float> GetOutputs();

private:
	float Activate(float input);

	int yeet = 1;
};


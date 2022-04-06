#pragma once
#include "Constants.h"

class PoolingLayer : public Layer
{
public:
	PoolingLayer(int inHei, int inWid, int noChannels, std::vector<float> in, int filHei, int filWid, int strHei, int strWid, bool isMax);
	void CalculateOutputs() override;
	void SetInputs(std::vector<float> in) override;
	std::vector<float> GetOutputs() override;

	/* Backpropagates the layer using the previous layer's bias costs and weights. */
	void BackPropagate(std::vector<float> previousLayerCosts) override;

	std::vector<float> GetInputCosts() override;

	/* Resets the weight and biases costs. */
	void ResetValues() override;

private:
	int inputHeight;
	int inputWidth;
	int numChannels;
	int filterHeight;
	int filterWidth;
	int strideHeight;
	int strideWidth;
	bool isMaxPooling;

	std::vector<std::vector<std::vector<float>>> inputImage;
	std::vector<std::vector<std::vector<float>>> outputImage;
	std::vector<std::vector<std::vector<float>>> lossInputImage;
	std::vector<float> lossInput;

	float MaxPool(int x, int y, int z);
	void BackPropagateMaxPool(int x, int y, int z, float previousLayerLoss);
	float AveragePool(int x, int y, int z);
};


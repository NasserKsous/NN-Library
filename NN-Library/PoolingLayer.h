#pragma once
#include "Constants.h"

class PoolingLayer : public Layer
{
public:
	PoolingLayer(int inHei, int inWid, int noChannels, std::vector<float> in, int filHei, int filWid, int strHei, int strWid, bool isMax);
	void CalculateOutputs() override;
	void SetInputs(std::vector<float> in) override;
	std::vector<float> GetOutputs() override;


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

	float MaxPool(int x, int y, int z);
	float AveragePool(int x, int y, int z);
};


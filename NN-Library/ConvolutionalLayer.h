#pragma once
#include "Constants.h"

class ConvolutionalLayer : public Layer
{
public:
	ConvolutionalLayer(int inHei, int inWid, std::vector<float> in, int filHei, int filWid, std::vector<float> weights, int padHei, int padWid, int strHei, int strWid);

	void CalculateOutputs() override;
	void SetInputs(std::vector<float> in);

private:
	int inputHeight;
	int inputWidth;
	int filterHeight;
	int filterWidth;
	int paddingHeight;
	int paddingWidth;
	int strideHeight;
	int strideWidth;

	std::vector<std::vector<float>> inputImage;
	std::vector<std::vector<float>> outputImage;
	std::vector<std::vector<float>> filter;
};

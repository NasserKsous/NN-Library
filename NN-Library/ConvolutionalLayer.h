#pragma once
#include "Constants.h"
#include "ActivationFunctions.h"

class ConvolutionalLayer : public Layer
{
public:
	ConvolutionalLayer(int inHei, int inWid, int noChannels, std::vector<float> in, int filHei, int filWid, std::vector<float> wei, int strHei, int strWid, bool pad, ACTIVATION actType);

	void CalculateOutputs() override;
	void SetInputs(std::vector<float> in);
	std::vector<float> GetOutputs();

private:
	int inputHeight;
	int inputWidth;
	int numChannels;
	int filterHeight;
	int filterWidth;
	int paddingHeight;
	int paddingWidth;
	int strideHeight;
	int strideWidth;

	std::vector<std::vector<std::vector<float>>> inputImage;
	std::vector<std::vector<std::vector<float>>> outputImage;
	std::vector<std::vector<std::vector<float>>> filter;

	bool hasPadding = false;
};


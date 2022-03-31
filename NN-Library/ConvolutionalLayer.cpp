#include "ConvolutionalLayer.h"
#include <assert.h>

ConvolutionalLayer::ConvolutionalLayer(int inHei, int inWid, std::vector<float> in, int filHei, int filWid, std::vector<float> wei, int strHei, int strWid, bool pad, ACTIVATION actType)
{
	inputHeight = inHei;
	inputWidth = inWid;
	inputs = in;
	filterHeight = filHei;
	filterWidth = filWid;
	weights = wei;
	hasPadding = pad;
	strideHeight = strHei;
	strideWidth = strWid;
	activation = actType;
	layerType = LAYER_TYPE::CONVOLUTIONAL;

	assert((int)in.size() == inputHeight * inputWidth && "Input is not the correct size");
	assert((int)wei.size() == filHei * filWid && "Filter is not the correct size");

	std::vector<float> inputRow;

	if (hasPadding)
	{
		std::vector<float> paddingRow(inputWidth + 2, 0.0f);
		inputImage.push_back(paddingRow);
	}

	for (int heightIndex = 0; heightIndex < inputHeight; ++heightIndex)
	{
		if (hasPadding)
		{
			inputRow.push_back(0.0f);
		}

		for (int widthIndex = 0; widthIndex < inputWidth; ++widthIndex)
		{
			inputRow.push_back(inputs[(heightIndex*inputWidth) + widthIndex]);
		}
		if (hasPadding)
		{
			inputRow.push_back(0.0f);
		}
		inputImage.push_back(inputRow);
		inputRow.clear();
	}
	if (hasPadding)
	{
		std::vector<float> paddingRow(inputWidth + 2, 0.0f);
		inputImage.push_back(paddingRow);

		inputWidth += 2;
		inputHeight += 2;
	}

	std::vector<float> filterRow;
	for (int heightIndex = 0; heightIndex < filterHeight; ++heightIndex)
	{
		for (int widthIndex = 0; widthIndex < filterWidth; ++widthIndex)
		{
			filterRow.push_back(weights[(heightIndex * filterWidth) + widthIndex]);
		}
		filter.push_back(filterRow);
		filterRow.clear();
	}
}

void ConvolutionalLayer::CalculateOutputs()
{
	outputImage.clear();
	outputs.clear();

	int halfFilterHeight = (int)filterHeight / 2;
	int halfFilterWidth = (int)filterWidth / 2;
	
	int maxHeight = inputHeight - halfFilterHeight - 1;
	int maxWight = inputWidth - halfFilterWidth - 1;

	std::vector<float> outputRow;

	for (int centreY = halfFilterHeight; centreY <= maxHeight; centreY += strideHeight)
	{
		for (int centreX = halfFilterWidth; centreX <= maxWight; centreX += strideWidth)
		{
			float output = 0.0f;

			for (int heightIndex = 0; heightIndex < filterHeight; ++heightIndex)
			{
				for (int widthIndex = 0; widthIndex < filterHeight; ++widthIndex)
				{
					output += filter[heightIndex][widthIndex] * inputImage[centreY + (heightIndex - halfFilterHeight)][centreX + (widthIndex - halfFilterWidth)];
				}
			}
			output = Activate(output, activation);
			outputs.push_back(output);
			outputRow.push_back(output);	
		}
		outputImage.push_back(outputRow);
		outputRow.clear();
	}
}

void ConvolutionalLayer::SetInputs(std::vector<float> in)
{
	assert((int)in.size() == inputHeight * inputWidth && "Input is not the correct size");
	
	inputs = in;
	inputImage.clear();
	std::vector<float> inputRow;
	for (int heightIndex = 0; heightIndex < inputHeight; ++heightIndex)
	{
		for (int widthIndex = 0; widthIndex < inputWidth; ++widthIndex)
		{
			inputRow.push_back(inputs[(heightIndex * inputWidth) + widthIndex]);
		}
		inputImage.push_back(inputRow);
		inputRow.clear();
	}
}

std::vector<float> ConvolutionalLayer::GetOutputs()
{
	return outputs;
}

#include "ConvolutionalLayer.h"

ConvolutionalLayer::ConvolutionalLayer(int inHei, int inWid, std::vector<float> in, int filHei, int filWid, std::vector<float> wei, int padHei, int padWid, int strHei, int strWid)
{
	inputHeight = inHei;
	inputWidth = inWid;
	inputs = in;
	filterHeight = filHei;
	filterWidth = filWid;
	weights = wei;
	paddingHeight = padHei;
	paddingWidth = padWid;
	strideHeight = strHei;
	strideWidth = strWid;

	if ((int)in.size() == inHei * inWid)
	{
		std::vector<float> inputRow;
		for (int heightIndex = 0; heightIndex < inputHeight; ++heightIndex)
		{
			for (int widthIndex = 0; widthIndex < inputWidth; ++widthIndex)
			{
				inputRow.push_back(inputs[(heightIndex*inputWidth) + widthIndex]);
			}
			inputImage.push_back(inputRow);
			inputRow.clear();
		}
	}

	if ((int)wei.size() == filHei * filWid)
	{
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
}

void ConvolutionalLayer::CalculateOutputs()
{
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
			outputRow.push_back(output);	
		}
		outputImage.push_back(outputRow);
		outputRow.clear();
	}
}

void ConvolutionalLayer::SetInputs(std::vector<float> in)
{
	if ((int)in.size() == inputHeight * inputWidth)
	{
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
}
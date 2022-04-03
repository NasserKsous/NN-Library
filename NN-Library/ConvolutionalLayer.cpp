#include "ConvolutionalLayer.h"
#include <assert.h>

ConvolutionalLayer::ConvolutionalLayer(int inHei, int inWid, int noChannels, std::vector<float> in, int filHei, int filWid, std::vector<float> wei, int strHei, int strWid, bool pad, ACTIVATION actType)
{
	inputHeight = inHei;
	inputWidth = inWid;
	numChannels = noChannels;
	inputs = in;
	filterHeight = filHei;
	filterWidth = filWid;
	weights = wei;
	hasPadding = pad;
	strideHeight = strHei;
	strideWidth = strWid;
	activation = actType;
	layerType = LAYER_TYPE::CONVOLUTIONAL;

	assert((int)in.size() == inputHeight * inputWidth * numChannels && "Input is not the correct size");
	assert((int)wei.size() == filterHeight * filterWidth * numChannels && "Filter is not the correct size");

	std::vector<std::vector<float>> inputChannel;
	std::vector<float> inputRow;

	for (int channelIndex = 0; channelIndex < numChannels; ++channelIndex)
	{
		if (hasPadding)
		{
			std::vector<float> paddingRow(inputWidth + 2, 0.0f);
			inputChannel.push_back(paddingRow);
		}

		for (int heightIndex = 0; heightIndex < inputHeight; ++heightIndex)
		{
			if (hasPadding)
			{
				inputRow.push_back(0.0f);
			}

			for (int widthIndex = 0; widthIndex < inputWidth; ++widthIndex)
			{
				inputRow.push_back(inputs[channelIndex * (inputWidth * inputHeight) + (heightIndex * inputWidth) + widthIndex]);
			}
			if (hasPadding)
			{
				inputRow.push_back(0.0f);
			}
			inputChannel.push_back(inputRow);
			inputRow.clear();
		}
		if (hasPadding)
		{
			std::vector<float> paddingRow(inputWidth + 2, 0.0f);
			inputChannel.push_back(paddingRow);

			inputWidth += 2;
			inputHeight += 2;
		}
		inputImage.push_back(inputChannel);
		inputChannel.clear();
	}

	std::vector<std::vector<float>> filterChannel;
	std::vector<float> filterRow;
	for (int channelIndex = 0; channelIndex < numChannels; ++channelIndex)
	{
		for (int heightIndex = 0; heightIndex < filterHeight; ++heightIndex)
		{
			for (int widthIndex = 0; widthIndex < filterWidth; ++widthIndex)
			{
				filterRow.push_back(weights[(heightIndex * filterWidth) + widthIndex]);
			}
			filterChannel.push_back(filterRow);
			filterRow.clear();
		}
		filter.push_back(filterChannel);
		filterChannel.clear();
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

	std::vector<std::vector<float>> outputChannel;
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
					for (int channelIndex = 0; channelIndex < numChannels; ++channelIndex)
					{
						output += filter[channelIndex][heightIndex][widthIndex] * inputImage[channelIndex][centreY + (heightIndex - halfFilterHeight)][centreX + (widthIndex - halfFilterWidth)];
					}
				}
			}
			output = Activate(output, activation);
			outputs.push_back(output);
			outputRow.push_back(output);
		}
		outputChannel.push_back(outputRow);
		outputRow.clear();
	}
	outputImage.push_back(outputChannel);
	outputChannel.clear();
}

void ConvolutionalLayer::SetInputs(std::vector<float> in)
{
	assert((int)in.size() == inputHeight * inputWidth * numChannels && "Input is not the correct size");
	
	inputs = in;
	inputImage.clear();
	std::vector<std::vector<float>> inputChannel;
	std::vector<float> inputRow;

	for (int channelIndex = 0; channelIndex < numChannels; ++channelIndex)
	{
		for (int heightIndex = 0; heightIndex < inputHeight; ++heightIndex)
		{
			for (int widthIndex = 0; widthIndex < inputWidth; ++widthIndex)
			{
				inputRow.push_back(inputs[channelIndex * (inputWidth * inputHeight) + (heightIndex * inputWidth) + widthIndex]);
			}
			inputChannel.push_back(inputRow);
			inputRow.clear();
		}
		inputImage.push_back(inputChannel);
		inputChannel.clear();
	}
}

std::vector<float> ConvolutionalLayer::GetOutputs()
{
	return outputs;
}

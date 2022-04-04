#include "PoolingLayer.h"
#include <assert.h>
#include <math.h>

PoolingLayer::PoolingLayer(int inHei, int inWid, int noChannels, std::vector<float> in, int filHei, int filWid, int strHei, int strWid, bool isMax)
{
	inputHeight = inHei;
	inputWidth = inWid;
	inputs = in;
	numChannels = noChannels;
	filterHeight = filHei;
	filterWidth = filWid;
	strideHeight = strHei;
	strideWidth = strWid;
	isMaxPooling = isMax;
	layerType = LAYER_TYPE::POOLING;

	assert((int)in.size() == inputHeight * inputWidth * numChannels && "Input is not the correct size");

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

void PoolingLayer::CalculateOutputs()
{
	std::vector<std::vector<float>> outputChannel;
	std::vector<float> outputRow;

	float output = 0.0f;

	int maxHeight = inputHeight - filterHeight;
	int maxWidth = inputWidth - filterWidth;

	for (int channelIndex = 0; channelIndex < numChannels; ++channelIndex)
	{
		for (int topLeftY = 0; topLeftY <= maxHeight; topLeftY += strideHeight)
		{
			for (int topLeftX = 0; topLeftX <= maxWidth; topLeftX += strideWidth)
			{
				output = 0.0f;

				if (isMaxPooling)
				{
					output = MaxPool(topLeftX, topLeftY, channelIndex);
				}
				else
				{
					output = AveragePool(topLeftX, topLeftY, channelIndex);
				}
				outputs.push_back(output);
				outputRow.push_back(output);
			}
			outputChannel.push_back(outputRow);
			outputRow.clear();
		}
		outputImage.push_back(outputChannel);
		outputChannel.clear();
	}
}

void PoolingLayer::SetInputs(std::vector<float> in)
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

std::vector<float> PoolingLayer::GetOutputs()
{
	return outputs;
}

float PoolingLayer::MaxPool(int x, int y, int z)
{
	float output = -INFINITY;

	for (int inputY = 0; inputY < filterHeight; ++inputY)
	{
		for (int inputX = 0; inputX < filterWidth; ++inputX)
		{
			output = std::max(output, inputImage[z][y + inputY][x + inputX]);
		}
	}

	return output;
}

float PoolingLayer::AveragePool(int x, int y, int z)
{
	float output = 0.0f;

	for (int inputY = 0; inputY < filterHeight; ++inputY)
	{
		for (int inputX = 0; inputX < filterWidth; ++inputX)
		{
			output += (output, inputImage[z][y + inputY][x + inputX]);
		}
	}

	output /= (filterHeight * filterWidth);

	return output;
}

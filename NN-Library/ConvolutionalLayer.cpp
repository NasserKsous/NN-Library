#include "ConvolutionalLayer.h"
#include <assert.h>

ConvolutionalLayer::ConvolutionalLayer(int inHei, int inWid, int noChannels, std::vector<Filter> wei, int strHei, int strWid, bool pad, ACTIVATION actType)
{
	inputHeight = inHei;
	inputWidth = inWid;
	numChannels = noChannels;
	numFilters = wei.size();
	hasPadding = pad;
	strideHeight = strHei;
	strideWidth = strWid;
	activation = actType;
	layerType = LAYER_TYPE::CONVOLUTIONAL;
	
	/*std::vector<std::vector<float>> inputChannel;
	std::vector<float> inputRow;*/

	/*for (int channelIndex = 0; channelIndex < numChannels; ++channelIndex)
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
	}*/
	
	// CHeck the filters have the same number of channels as the input image before adding them to the array of filters.
	for (Filter weight : wei)
	{
		assert((int)weight.channels == numChannels && "Filter is not the correct size");
		filters.push_back(weight);
	}
}

void ConvolutionalLayer::CalculateOutputs()
{
	outputImage.clear();
	outputs.clear();

	for (Filter filter : filters)
	{
		int maxHeight = inputHeight - (filter.height - 1);
		int maxWidth = inputWidth - (filter.width - 1);

		std::vector<std::vector<float>> outputChannel;
		std::vector<float> outputRow;

		for (int inputY = 0; inputY < maxHeight; inputY += strideHeight)
		{
			for (int inputX = 0; inputX < maxWidth; inputX += strideWidth)
			{
				float output = 0.0f;

				for (int heightIndex = 0; heightIndex < filter.height; ++heightIndex)
				{
					for (int widthIndex = 0; widthIndex < filter.width; ++widthIndex)
					{
						for (int channelIndex = 0; channelIndex < numChannels; ++channelIndex)
						{
							output += filter.values[channelIndex][heightIndex][widthIndex] * inputImage[channelIndex][inputY + (heightIndex)][inputX + (widthIndex)];
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
}

std::vector<float> ConvolutionalLayer::GetOutputs()
{
	return outputs;
}

std::vector<Filter> ConvolutionalLayer::GetFilters()
{
	return filters;
}

void ConvolutionalLayer::BackPropagate(std::vector<float> lossOfPreviousLayer, int lossHeight, int lossWidth, int lossChannels)
{

	std::vector<std::vector<std::vector<float>>> lossImage;
	std::vector<std::vector<float>> lossChannel;
	std::vector<float> lossRow;
	for (int channelIndex = 0; channelIndex < lossChannels; ++channelIndex)
	{
		for (int heightIndex = 0; heightIndex < lossHeight; ++heightIndex)
		{
			for (int widthIndex = 0; widthIndex < lossWidth; ++widthIndex)
			{
				lossRow.push_back(lossOfPreviousLayer[channelIndex * (lossHeight*lossWidth) + (heightIndex * lossWidth) + widthIndex]);
			}
			lossChannel.push_back(lossRow);
			lossRow.clear();
		}
		lossImage.push_back(lossChannel);
		lossChannel.clear();
	}

	int channelIndex = 0;
	std::vector<std::vector<float>> filterChannel;
	std::vector<float> filterRow;

	// Loss for filters
	for (Filter filter: filters)
	{
		int maxHeight = inputHeight - (lossHeight - 1);
		int maxWidth = inputWidth - (lossWidth -1);

		for (int inputY = 0; inputY < maxHeight; inputY += strideHeight)
		{
			for (int inputX = 0; inputX < maxWidth; inputX += strideWidth)
			{
				float output = 0.0f;

				for (int heightIndex = 0; heightIndex < lossHeight; ++heightIndex)
				{
					for (int widthIndex = 0; widthIndex < lossWidth; ++widthIndex)
					{
						output += lossImage[channelIndex][heightIndex][widthIndex] * inputImage[channelIndex][inputY + (heightIndex)][inputX + (widthIndex)];
					}
				}
				filterRow.push_back(output);
			}
			filterChannel.push_back(filterRow);
			filterRow.clear();
		}
		filter.lossValues.push_back(filterChannel);
		filterChannel.clear();

		++channelIndex;
	}


	//Loss for inputs
	std::vector<Filter> transposedFilters = filters;
	for (Filter filter : transposedFilters)
	{
		std::vector<std::vector<std::vector<float>>> tempValues = filter.values;

		for (int channelIndex = 0; channelIndex < filter.channels; ++channelIndex)
		{
			for (int heightIndex = 0; heightIndex < filter.height; ++heightIndex)
			{
				for (int widthIndex = 0; widthIndex < filter.width; ++widthIndex)
				{
					filter.values[channelIndex][heightIndex][widthIndex] = tempValues[numChannels - 1 - channelIndex][filter.height - 1 - heightIndex][filter.width - 1 - widthIndex];
				}
			}
		}


		for (int inputY = 0; inputY < lossHeight; inputY += strideHeight)
		{
			for (int inputX = 0; inputX < lossWidth; inputX += strideWidth)
			{
				float output = 0.0f;

				for (int heightIndex = 0; heightIndex < filter.height; ++heightIndex)
				{
					for (int widthIndex = 0; widthIndex < filter.width; ++widthIndex)
					{
						if (heightIndex >= lossHeight || widthIndex >= lossWidth)
							continue;
						output += filter.values[channelIndex][heightIndex][widthIndex] * inputImage[channelIndex][inputY + (heightIndex)][inputX + (widthIndex)];
					}
				}
				filterRow.push_back(output);
			}
			filterChannel.push_back(filterRow);
			filterRow.clear();
		}
		loosInputImage.push_back(filterChannel);
		filterChannel.clear();

		++channelIndex;
	}


	//Reference for backprop: https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c
}

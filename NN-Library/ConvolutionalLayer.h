#pragma once
#include "Constants.h"
#include "ActivationFunctions.h"

struct Filter
{
	int height;
	int width;
	int channels;
	std::vector<std::vector<std::vector<float>>> values;
	std::vector<std::vector<std::vector<float>>> lossValues;

	Filter(int hei, int wid, int cha, std::vector<float> wei)
	{
		height = hei;
		width = wid;
		channels = cha;

		std::vector<std::vector<float>> filterChannel;
		std::vector<float> filterRow;
		for (int channelIndex = 0; channelIndex < channels; ++channelIndex)
		{
			for (int heightIndex = 0; heightIndex < height; ++heightIndex)
			{
				for (int widthIndex = 0; widthIndex < width; ++widthIndex)
				{
					filterRow.push_back(wei[channelIndex * (width * height) + (heightIndex * width) + widthIndex]);
				}
				filterChannel.push_back(filterRow);
				filterRow.clear();
			}
			values.push_back(filterChannel);
			filterChannel.clear();
		}
	}
};

class ConvolutionalLayer : public Layer
{
public:
	ConvolutionalLayer(int inHei, int inWid, int noChannels, std::vector<Filter> wei, int strHei, int strWid, bool pad, ACTIVATION actType);

	void CalculateOutputs() override;
	void SetInputs(std::vector<float> in) override;
	std::vector<float> GetOutputs() override;
	std::vector<Filter> GetFilters();
	std::vector<float> GetInputCosts() override;
	std::vector<float> GetWeightCosts() override;
	void UpdateWeightsAndBiases(std::vector<float> expWeightsCosts, std::vector<float> expBiasesCosts, float learningRate) override;

	/* Resets the weight and biases costs. */
	void ResetValues() override;

	void BackPropagate(std::vector<float> lossOfPreviousLayer) override;

private:
	int inputHeight;
	int inputWidth;
	int numChannels;
	int numFilters;
	int strideHeight;
	int strideWidth;

	std::vector<std::vector<std::vector<float>>> inputImage;
	std::vector<std::vector<std::vector<float>>> outputImage;
	std::vector<float> lossInput;
	std::vector<float> lossWeight;
	std::vector<Filter> filters;

	bool hasPadding = false;
};


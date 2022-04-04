#pragma once
#include "Constants.h"
#include "ActivationFunctions.h"

struct Filter
{
	int height;
	int width;
	int channels;
	std::vector<std::vector<std::vector<float>>> values;

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
	ConvolutionalLayer(int inHei, int inWid, int noChannels, std::vector<float> in, std::vector<Filter> wei, int strHei, int strWid, bool pad, ACTIVATION actType);

	void CalculateOutputs() override;
	void SetInputs(std::vector<float> in);
	std::vector<float> GetOutputs();
	std::vector<Filter> GetFilters();

private:
	int inputHeight;
	int inputWidth;
	int numChannels;
	int numFilters;
	int paddingHeight;
	int paddingWidth;
	int strideHeight;
	int strideWidth;

	std::vector<std::vector<std::vector<float>>> inputImage;
	std::vector<std::vector<std::vector<float>>> outputImage;
	std::vector<Filter> filters;

	bool hasPadding = false;
};


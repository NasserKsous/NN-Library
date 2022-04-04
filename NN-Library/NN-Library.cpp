// NN-Library.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <stdlib.h>
#include <stdint.h>
#include <fstream>
#include <math.h>

#include "ActivationFunctions.h"
#include "ConnectedLayer.h"
#include "ConvolutionalLayer.h"
#include "NeuralNetwork.h"
#include "Constants.h"

#include "pbPlots.hpp"
#include "supportLib.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "mnist/mnist_reader_less.hpp"

//Reference for plotting graph: https://github.com/InductiveComputerScience/pbPlots/tree/v0.1.9.0/Cpp

//Reference for MNIST reading: https://github.com/wichtounet/mnist

//Reference for MNIST dataset: http://yann.lecun.com/exdb/mnist/

void PlotGraph(std::vector<double> x, std::vector<double> y, std::string fileName)
{
	bool success;
	StringReference* errorMessage = new StringReference();

	RGBABitmapImageReference* imageReference = CreateRGBABitmapImageReference();

	success = DrawScatterPlot(imageReference, 600, 400, &x, &y, errorMessage);

	if (success)
	{
		std::vector<double>* pngdata = ConvertToPNG(imageReference->image);
		WriteToFile(pngdata, fileName);
		DeleteImage(imageReference->image);
	}
	else
	{
		std::cerr << "Error: ";
		for (wchar_t c : *errorMessage->string)
		{
			std::wcerr << c;
		}
		std::cerr << std::endl;
	}
}

void SineWave()
{
	srand(time(NULL));

	std::vector<float> testInputs = { 0.0f };
	std::vector<float> testWeights;
	testWeights.clear();
	for (int i = 0; i < 10; ++i)
	{
		float upper = 1.0f / sqrtf(1);
		float lower = 0.0f;
		int randomInt = (rand() % 1000);
		float tempWeight = (lower + randomInt * (upper - lower)) / 1000.0f;
		testWeights.push_back(tempWeight);
	}
	std::vector<float> testBiases = { 0.5f };
	testBiases.clear();
	for (int i = 0; i < 10; ++i)
	{
		float tempBias = 0.01;
		testBiases.push_back(tempBias);
	}
	int testNodes = 10;
	ACTIVATION testActivation = ACTIVATION::SIGMOID;
	ConnectedLayer* cLayer = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);

	testInputs = { 0.0f };
	testWeights.clear();
	for (int i = 0; i < 25*testNodes; ++i)
	{
		float upper = 1.0f / sqrtf(testNodes);
		float lower = 0.0f;
		int randomInt = (rand() % 1000);
		float tempWeight = (lower + randomInt * (upper - lower)) / 1000.0f;
		testWeights.push_back(tempWeight);
	}
	testBiases.clear();
	for (int i = 0; i < 25; ++i)
	{
		float tempBias = 0.01f;
		testBiases.push_back(tempBias);
	}
	testNodes = 25;
	testActivation = ACTIVATION::SIGMOID;
	ConnectedLayer* cLayer2 = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);

	testInputs = { 0.0f };
	testWeights.clear();
	for (int i = 0; i < 25 * testNodes; ++i)
	{
		float upper = 1.0f / sqrtf(testNodes);
		float lower = 0.0f;
		int randomInt = (rand() % 1000);
		float tempWeight = (lower + randomInt * (upper - lower)) / 1000.0f;
		testWeights.push_back(tempWeight);
	}
	testBiases.clear();
	for (int i = 0; i < 25; ++i)
	{
		float tempBias = 0.01f;
		testBiases.push_back(tempBias);
	}
	testNodes = 25;
	testActivation = ACTIVATION::SIGMOID;
	ConnectedLayer* cLayer3 = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);
	
	testInputs = { 0.0f };
	testWeights.clear();
	for (int i = 0; i < testNodes; ++i)
	{
		float upper = 1.0f / sqrtf(testNodes);
		float lower = -(1.0f / sqrtf(testNodes));
		int randomInt = (rand() % 1000) - 500;
		float tempWeight = (lower + randomInt * (upper - lower)) / 1000.0f;
		testWeights.push_back(tempWeight);
	}
	testBiases = { 0.01f };
	testNodes = 1;
	testActivation = ACTIVATION::TANH;
	ConnectedLayer* cLayer4 = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);

	int sets = 400;
	NeuralNetwork* nn = new NeuralNetwork(sets);
	nn->AddLayer(cLayer);
	nn->AddLayer(cLayer2);
	nn->AddLayer(cLayer3);
	nn->AddLayer(cLayer4);

	const int iterations = 1000;
	std::vector<float> outputs;
	float cost = 0.0f;
	std::vector<double> outputCosts(iterations);
	std::vector<double> outputIterations(iterations);

	std::vector<float> expectedOutputs;

	testInputs.clear();
	expectedOutputs.clear();
	float tempIter = 180.0f / sets;
	float tempInput = 0.0f;
	for (int setIndex = 0; setIndex < sets; ++setIndex)
	{
		tempInput += tempIter;
		testInputs.push_back(tempInput);
		expectedOutputs.push_back(sinf(tempInput * (M_PI/180.0f)));

	}

	for (int i = 0; i < iterations; ++i)
	{
		cost = 0.0f;

		nn->TrainNetwork(testInputs, expectedOutputs);
		cost = nn->GetCost();

		outputCosts[i] = cost;
		outputIterations[i] = double(i + 1.0);
		outputs = nn->GetOutputs();
		std::cout << "\nCost = " << cost << "\n\n";
	}

	std::vector<float> input = { 0.0f };
	nn->SetInputs(input);
	nn->CalculateOutputs();
	outputs = nn->CalculateOutputs();

	std::cout << "Output = " << outputs[0] << "\n\n";
	
	input = { 90.0f };
	nn->SetInputs(input);
	nn->CalculateOutputs();
	outputs = nn->CalculateOutputs();

	std::cout << "Output = " << outputs[0] << "\n\n";
	
	input = { 180.0f };
	nn->SetInputs(input);
	nn->CalculateOutputs();
	outputs = nn->CalculateOutputs();

	std::cout << "Output = " << outputs[0] << "\n\n";

	PlotGraph(outputIterations, outputCosts, "SINE-CostOverTime.png");

	system("SINE-CostOverTime.png");
}

void XOR()
{
	std::vector<float> testInputs = { 0.0f, 0.0f };
	std::vector<float> testWeights = { 0.2f, 0.8f, 0.4f, 0.6f };
	std::vector<float> testBiases = { 0.5f, 0.6f };
	int testNodes = 2;
	ACTIVATION testActivation = ACTIVATION::SIGMOID;
	ConnectedLayer* cLayer = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);

	testInputs = { 0.0f, 0.0f };
	testWeights = { 0.4f, 0.45f };
	testBiases = { 0.6f };
	testNodes = 1;
	testActivation = ACTIVATION::SIGMOID;
	ConnectedLayer* cLayer2 = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);

	int sets = 4;
	NeuralNetwork* nn = new NeuralNetwork(sets);
	nn->AddLayer(cLayer);
	nn->AddLayer(cLayer2);


	const int iterations = 10000;
	std::vector<float> outputs;
	float cost = 0.0f;
	std::vector<double> outputCosts(iterations);
	std::vector<double> outputIterations(iterations);

	std::vector<float> expectedOutputs;

	for (int i = 0; i < iterations; ++i)
	{
		cost = 0.0f;

		testInputs = { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f };
		expectedOutputs = { 0.0f, 1.0f, 1.0f, 0.0f };
		nn->TrainNetwork(testInputs, expectedOutputs);
		cost = nn->GetCost();

		outputCosts[i] = cost;
		outputIterations[i] = double(i + 1.0);
		outputs = nn->GetOutputs();
		std::cout << "\nCost = " << cost << "\n\n";
	}

	std::vector<float> input = { 0.0f, 0.0f };
	nn->SetInputs(input);
	outputs = nn->CalculateOutputs();

	std::cout << "Output = " << outputs[0] << "\n\n";

	input = { 0.0f, 1.0f };
	nn->SetInputs(input);
	nn->CalculateOutputs();
	outputs = nn->CalculateOutputs();

	std::cout << "Output = " << outputs[0] << "\n\n";

	input = { 1.0f, 0.0f };
	nn->SetInputs(input);
	nn->CalculateOutputs();
	outputs = nn->CalculateOutputs();

	std::cout << "Output = " << outputs[0] << "\n\n";

	input = { 1.0f, 1.0f };
	nn->SetInputs(input);
	nn->CalculateOutputs();
	outputs = nn->CalculateOutputs();

	std::cout << "Output = " << outputs[0] << "\n\n";

	PlotGraph(outputIterations, outputCosts, "XOR-CostOverTime.png");

	system("XOR-CostOverTime.png");
}

int main()
{
	//XOR();

	//SineWave();

	/*int width, height, bpp;

	float* rgb_image = stbi_loadf("testImage.png", &width, &height, &bpp, 3);*/

	auto dataset = mnist::read_dataset();

	std::cout << "Number of training images = " << dataset.training_images.size() << std::endl;
	std::cout << "Number of training labels = " << dataset.training_labels.size() << std::endl;
	std::cout << "Number of test images = " << dataset.test_images.size() << std::endl;
	std::cout << "Number of test labels = " << dataset.test_labels.size() << std::endl;

	for (int i = 0; i < 28; ++i)
	{
		for (int j = 0; j < 28; ++j)
		{
			std::cout << (float)dataset.test_images[0][i * 28 + j] << "   ";
		}
		std::cout << "\n";
	}

	std::vector<float> input = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 
								 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 
								 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 
								 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 
								 20.0f, 21.0f, 22.0f, 23.0f, 24.0f };
	
	
	std::vector<float> weight = { 0.0f, 1.0f, 0.0f,  
								  0.0f, 1.0f, 0.0f,
								  0.0f, 1.0f, 0.0f };

	std::vector<Filter> filters = { Filter(3, 3, 1, weight) };

	ConvolutionalLayer* convLayer = new ConvolutionalLayer(5, 5, 1, filters, 1, 1, true, ACTIVATION::RELU);
	convLayer->SetInputs(input);
	convLayer->CalculateOutputs();
	std::vector<float> outputs = convLayer->GetOutputs();
	for (int heightIndex = 0; heightIndex < 5; ++heightIndex)
	{
		for (int widthIndex = 0; widthIndex < 5; ++widthIndex)
		{
			std::cout << outputs[heightIndex * 5 + widthIndex] << ", ";
		}
		std::cout << "\n";
	}
	return 0;
}
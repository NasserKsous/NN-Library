// NN-Library.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <stdlib.h>
#include <stdint.h>
#include <fstream>
#include <math.h>
#include <chrono>

#include "ActivationFunctions.h"
#include "ConnectedLayer.h"
#include "ConvolutionalLayer.h"
#include "PoolingLayer.h"
#include "NeuralNetwork.h"
#include "Constants.h"

#include "pbPlots.hpp"
#include "supportLib.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "mnist/mnist_reader_less.hpp"

#include <random>

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


	const int iterations = 500000;
	std::vector<float> outputs;
	float cost = 0.0f;
	std::vector<double> outputCosts(iterations);
	std::vector<double> outputIterations(iterations);

	std::vector<float> expectedOutputs;

	auto start = std::chrono::steady_clock::now();

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

	auto end = std::chrono::steady_clock::now();

	std::vector<float> input = { 0.0f, 0.0f };
	nn->SetInputs(input);
	outputs = nn->CalculateOutputs();

	std::cout << "Inputs: " << input[0] << ", " << input[1] << " and Output = " << outputs[0] << "\n\n";

	input = { 0.0f, 1.0f };
	nn->SetInputs(input);
	nn->CalculateOutputs();
	outputs = nn->CalculateOutputs();

	std::cout << "Inputs: " << input[0] << ", " << input[1] << " and Output = " << outputs[0] << "\n\n";

	input = { 1.0f, 0.0f };
	nn->SetInputs(input);
	nn->CalculateOutputs();
	outputs = nn->CalculateOutputs();

	std::cout << "Inputs: " << input[0] << ", " << input[1] << " and Output = " << outputs[0] << "\n\n";

	input = { 1.0f, 1.0f };
	nn->SetInputs(input);
	nn->CalculateOutputs();
	outputs = nn->CalculateOutputs();

	std::cout << "Inputs: " << input[0] << ", " << input[1] << " and Output = " << outputs[0] << "\n\n";

	std::cout << "Time taken to train: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds\n\n";

	PlotGraph(outputIterations, outputCosts, "XOR-CostOverTime.png");

	system("XOR-CostOverTime.png");
}

void MNIST()
{
	/*int width, height, bpp;

	float* rgb_image = stbi_loadf("testImage.png", &width, &height, &bpp, 3);*/

	auto dataset = mnist::read_dataset();

	std::cout << "Number of training images = " << dataset.training_images.size() << std::endl;
	std::cout << "Number of training labels = " << dataset.training_labels.size() << std::endl;
	std::cout << "Number of test images = " << dataset.test_images.size() << std::endl;
	std::cout << "Number of test labels = " << dataset.test_labels.size() << std::endl;
	std::cout << "\n\n";
	/*for (int i = 0; i < 28; ++i)
	{
		for (int j = 0; j < 28; ++j)
		{
			std::cout << (float)dataset.test_images[0][i * 28 + j] << "   ";
		}
		std::cout << "\n";
	}*/

	int iterations = 1000;
	int batchSize = 150;
	NeuralNetwork* nn = new NeuralNetwork(batchSize);
	srand(time(NULL));
	std::vector<float> weights;
	std::vector<Filter> filters;
	double temp = 2.0 / 676.0;
	double variance = sqrt(temp);
	std::normal_distribution<double> distribution(0.0, variance);
	std::default_random_engine gen;
	float randomWeight = (float)distribution(gen);
	float randomWeight2 = (float)distribution(gen);

	for (int j = 0; j < 6; ++j)
	{
		for (int i = 0; i < 25; ++i)
		{
			weights.push_back((float)distribution(gen));
		}
		Filter filter(5, 5, 1, weights);
		filters.push_back(filter);
		weights.clear();
	}

	ConvolutionalLayer* convLayer = new ConvolutionalLayer(28, 28, 1, filters, 1, 1, false, ACTIVATION::RELU);

	std::vector<float> tempInputs(24 * 24 * 6);
	PoolingLayer* poolLayer = new PoolingLayer(24, 24, 6, tempInputs, 3, 3, 3, 3, true);

	std::vector<float> tempInputs2(8*8*6);
	for (int i = 0; i < 8*8*6*10; ++i)
	{
		weights.push_back((float)distribution(gen));
	}
	std::vector<float> biases(10);

	ConnectedLayer* layer = new ConnectedLayer(tempInputs2, weights, biases, 10, ACTIVATION::SOFTMAX);

	std::vector<float> tempInputs3(784);
	weights.clear();
	for (int i = 0; i < 784 * 10; ++i)
	{
		weights.push_back((float)distribution(gen));
	}
	std::vector<float> biases2(10);

	//ConnectedLayer* layer2 = new ConnectedLayer(tempInputs3, weights, biases2, 10, ACTIVATION::SOFTMAX);

	nn->AddLayer(convLayer);
	nn->AddLayer(poolLayer);
	nn->AddLayer(layer);

	//nn->AddLayer(layer2);

	std::vector<float> inputs;
	int sizeOfImages = dataset.training_images[0].size();
	int numberOfImages = dataset.training_images.size();
	std::vector<float> expectedOutputs;
	std::vector<double> outputCosts(iterations);
	std::vector<double> outputIterations(iterations);

	auto start = std::chrono::steady_clock::now();

	for (int i = 0; i < iterations; ++i)
	{
		float cost = 0.0f;

		for (int i = 0; i < batchSize; ++i)
		{
			int randomIndex = rand() % numberOfImages;
			for (int j = 0; j < sizeOfImages; ++j)
			{
				inputs.push_back(dataset.training_images[randomIndex][j] / 255.0f);
			}
			int expectedOutput = dataset.training_labels[randomIndex];
			for (int j = 0; j < 10; ++j)
			{
				if (j == expectedOutput)
				{
					expectedOutputs.push_back(1.0f);
				}
				else
				{
					expectedOutputs.push_back(0.0f);
				}
			}
		}

		nn->TrainNetwork(inputs, expectedOutputs);
		cost = nn->GetCost();

		outputCosts[i] = cost;
		outputIterations[i] = double(i + 1.0);
		std::cout << "\n Iteration = " << i + 1 << ", Cost = " << cost << "\n\n";

		expectedOutputs.clear();
		inputs.clear();
	}

	auto end = std::chrono::steady_clock::now();

	for (int j = 0; j < sizeOfImages; ++j)
	{
		inputs.push_back(dataset.test_images[0][j] / 255.0f);
	}

	int expectedOutput = dataset.test_labels[0];
	for (int j = 0; j < 10; ++j)
	{
		if (j == expectedOutput)
		{
			expectedOutputs.push_back(1.0f);
		}
		else
		{
			expectedOutputs.push_back(0.0f);
		}
	}
	nn->SetInputs(inputs);
	nn->CalculateOutputs();
	std::vector<float> outputs = nn->CalculateOutputs();
	int out = 0;
	for (int i = 0; i < outputs.size(); ++i)
	{
		if (outputs[i] > 0.9f)
		{
			out = i;
			break;
		}
	}
	std::cout << "Output = " << out << ", Expected Output = " << expectedOutput << "\n";

	numberOfImages = dataset.test_images.size();
	int correctCount = 0;
	int testIterations = 1000;
	for (int k = 0; k < testIterations; ++k)
	{
		int randomIndex = rand() % numberOfImages;

		inputs.clear();
		for (int j = 0; j < sizeOfImages; ++j)
		{
			inputs.push_back(dataset.test_images[randomIndex][j] / 255.0f);
		}

		expectedOutput = dataset.test_labels[randomIndex];
		for (int j = 0; j < 10; ++j)
		{
			if (j == expectedOutput)
			{
				expectedOutputs.push_back(1.0f);
			}
			else
			{
				expectedOutputs.push_back(0.0f);
			}
		}
		nn->SetInputs(inputs);
		nn->CalculateOutputs();
		outputs = nn->CalculateOutputs();
		int max = 0;
		for (int i = 1; i < outputs.size(); ++i)
		{
			if (outputs[i] > outputs[max])
			{
				max = i;
			}
		}
		out = max;
		if (out == expectedOutput)
			++correctCount;
		std::cout << "Output = " << out << ", Expected Output = " << expectedOutput << "\n";
	}

	std::cout << "\nNumber of correct tests = " << correctCount << "/" << testIterations << "\n";

	std::cout << "Time taken to train: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds\n\n";

	PlotGraph(outputIterations, outputCosts, "MNIST-CostOverTime.png");

	system("MNIST-CostOverTime.png");
}

int main()
{
	int choice = 0;

	std::cout << "1 for XOR or 2 for MNIST:";
	std::cin >> choice;

	switch (choice)
	{
	case 1:
		XOR();
		break;
	case 2:
		MNIST();
		break;
	default:
		std::cout << "Nothing has been selected";
		break;
	}
	
	return 0;
}
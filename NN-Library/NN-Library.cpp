// NN-Library.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <stdlib.h>
#include "ActivationFunctions.h"
#include "ConnectedLayer.h"
#include "NeuralNetwork.h"
#include "Constants.h"

#include "pbPlots.hpp"
#include "supportLib.hpp"

#include <math.h>

//Reference for plotting graph: https://github.com/InductiveComputerScience/pbPlots/tree/v0.1.9.0/Cpp

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
	for (int i = 0; i < 128; ++i)
	{
		int randomInt = (rand() % 200) - 100;
		float tempWeight = (float)randomInt / 100.0f;
		testWeights.push_back(tempWeight);
	}
	std::vector<float> testBiases = { 0.5f, 0.6f, 1.0f, 5.0f, 4.0f, -2.0f, -4.0f, -3.0f, -2.6f, 3.3f, 0.5f, 0.6f, 1.0f, 5.0f, 4.0f, -2.0f, -4.0f, -3.0f, -2.6f, 3.3f };
	testBiases.clear();
	for (int i = 0; i < 128; ++i)
	{
		int randomInt = (rand() % 200) - 100;
		float tempBias = (float)randomInt / 100.0f;
		testBiases.push_back(tempBias);
	}
	int testNodes = 128;
	ACTIVATION testActivation = ACTIVATION::SIGMOID;
	ConnectedLayer* cLayer = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);

	/*testInputs = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	testWeights.clear();
	for (int i = 0; i < 200; ++i)
	{
		int randomInt = (rand() % 200) - 100;
		float tempWeight = (float)randomInt / 100.0f;
		testWeights.push_back(tempWeight);
	}
	testBiases = { 0.5f, 0.6f, 1.0f, 5.0f, 4.0f, -2.0f, -4.0f, -3.0f, -2.6f, 3.3f };
	testNodes = 10;
	testActivation = ACTIVATION::SIGMOID;
	ConnectedLayer* cLayer2 = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);*/

	testInputs = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	testWeights.clear();
	for (int i = 0; i < 128; ++i)
	{
		int randomInt = (rand() % 200) - 100;
		float tempWeight = (float)randomInt / 100.0f;
		testWeights.push_back(tempWeight);
	}
	testBiases = { 0.6f };
	testNodes = 1;
	testActivation = ACTIVATION::SIGMOID;
	ConnectedLayer* cLayer3 = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);

	int sets = 11;
	NeuralNetwork* nn = new NeuralNetwork(sets);
	nn->AddLayer(cLayer);
	//nn->AddLayer(cLayer2);
	nn->AddLayer(cLayer3);

	const int iterations = 10000;
	std::vector<float> outputs;
	float cost = 0.0f;
	std::vector<double> outputCosts(iterations);
	std::vector<double> outputIterations(iterations);

	std::vector<float> expectedOutputs;

	testInputs.clear();
	expectedOutputs.clear();
	for (int setIndex = 0; setIndex < sets; ++setIndex)
	{
		testInputs.push_back(M_PI * 2 / (setIndex + 1.0f));
	}
	
	testInputs = { 0.0f, 45.0f, 90.0f, 135.0f, 180.0f, 217.0f, 225.0f, 270.0f, 279.0f, 315.0f, 360.0f };
	expectedOutputs = { 0.0f, 0.7071f, 1.0f, 0.7071f, 0.0f, -0.6018f, -0.7071f, -1.0f, -0.9877f, -0.7071f, 0.0f };

	for (int i = 0; i < iterations; ++i)
	{
		cost = 0.0f;

		nn->TrainNetwork(testInputs, expectedOutputs);
		cost = nn->GetCost();

		cost /= sets;
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
	
	input = { 270.0f };
	nn->SetInputs(input);
	nn->CalculateOutputs();
	outputs = nn->CalculateOutputs();

	std::cout << "Output = " << outputs[0] << "\n\n";
	
	input = { 360.0f };
	nn->SetInputs(input);
	nn->CalculateOutputs();
	outputs = nn->CalculateOutputs();

	std::cout << "Output = " << outputs[0] << "\n\n";

	PlotGraph(outputIterations, outputCosts, "XOR-CostOverTime.png");

	system("XOR-CostOverTime.png");
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

		cost /= sets;
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

	SineWave();

	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file

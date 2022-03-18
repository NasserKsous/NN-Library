// NN-Library.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "ActivationFunctions.h"
#include "ConnectedLayer.h"
#include "NeuralNetwork.h"
#include "Constants.h"

#include "pbPlots.hpp"
#include "supportLib.hpp"

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

int main()
{
	std::vector<float> testInputs = { 0.05f, 0.1f };
	std::vector<float> testWeights = { 0.15f, 0.2f, 0.25f, 0.3f };
	std::vector<float> testBiases = { 0.35f, 0.35f };
	std::vector<float> expectedOutputs = { 0.01f, 0.99f };
	int testNodes = 2;
	ACTIVATION testActivation = ACTIVATION::SIGMOID;
	ConnectedLayer* cLayer = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);
	
	testInputs = { 0.05f, 0.1f };
	testWeights = { 0.4f, 0.45f, 0.5f, 0.55f };
	testBiases = { 0.6f, 0.6f };
	testNodes = 2;
	testActivation = ACTIVATION::SIGMOID;
	ConnectedLayer* cLayer2 = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);

	NeuralNetwork* nn = new NeuralNetwork();
	nn->AddLayer(cLayer);
	nn->AddLayer(cLayer2);


	const int iterations = 1000;
	std::vector<float> outputs;
	float cost = 0.0f;
	std::vector<double> outputCosts;
	std::vector<double> outputIterations;
	std::vector<double> outputOutputs;
	std::vector<double> outputOutputs2;

	for (int i = 0; i < iterations; ++i)
	{
		nn->CalculateOutputs();
		nn->BackPropagate(expectedOutputs);
		cost = nn->GetCost();
		outputCosts.push_back(cost);
		outputIterations.push_back(double(i + 1));
		outputs = nn->GetOutputs();

		std::cout << "Output 0 = " << outputs[0] << "\nOutput 1 = " << outputs[1] << "\nCost = " << cost << "\n\n";
		outputOutputs.push_back(outputs[0]);
		outputOutputs2.push_back(outputs[1]);
	}

	
	PlotGraph(outputIterations, outputCosts, "OutputCostOverTime.png");
	PlotGraph(outputIterations, outputOutputs, "OutputOneOverTime.png");
	PlotGraph(outputIterations, outputOutputs2, "OutputTwoOverTime.png");

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

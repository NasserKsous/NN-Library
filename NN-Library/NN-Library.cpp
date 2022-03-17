// NN-Library.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "ActivationFunctions.h"
#include "ConnectedLayer.h"
#include "NeuralNetwork.h"
#include "Constants.h"

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

	std::vector<float> outputs;
	float cost = 0.0f;

	for (int i = 0; i < 100; ++i)
	{
		nn->CalculateOutputs();
		nn->BackPropagate(expectedOutputs);
		cost = nn->GetCost();
		outputs = nn->GetOutputs();

		std::cout << "Output 0 = " << outputs[0] << "\nOutput 1 = " << outputs[1] << "\nCost = " << cost << "\n\n";
	}

	
	std::vector<Layer*> testNN = nn->GetNetwork();
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

// NN-Library.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "ActivationFunctions.h"
#include "ConnectedLayer.h"
#include "NeuralNetwork.h"
#include "Constants.h"

int main()
{
	std::vector<float> testInputs = { 34.5f, 23.2f, 0.23f };
	std::vector<float> testWeights = { 1.2f, 0.04f, -25.0f, 3.1f, 35.06f, 0.0f, -12.22f, 1.74f, -67.0f };
	std::vector<float> testBiases = { 32.0f, 54.0f, -1.0f };
	int testNodes = 3;
	ACTIVATION testActivation = ACTIVATION::RELU;
	ConnectedLayer* cLayer = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);

	testWeights = { 7.0f, 0.4f, -5.0f, -23.1f, 0.06f, 24.0f };
	testBiases = { 12.0f, 4.0f };
	testNodes = 2;
	testActivation = ACTIVATION::RELU;
	ConnectedLayer* cLayer2 = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);

	std::vector<float> testOutputs = { 881.7828f, 0.0f, 0.0f };

	NeuralNetwork* nn = new NeuralNetwork();
	nn->AddLayer(cLayer);
	nn->AddLayer(cLayer2);
	nn->CalculateOutputs();
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

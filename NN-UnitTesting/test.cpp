#include "pch.h"
#include "../NN-Library/ActivationFunctions.h"
#include "../NN-Library/ConnectedLayer.h"
#include "../NN-Library/ConnectedLayer.cpp"

namespace NeuralNetworkLibrary
{
	
	static float Round(float a)
	{
		return (a > 0) ? ::floor(a + 0.5f) : ::ceil(a - 0.5f);
	}
	static float Round(float a, int places)
	{
		const float shift = pow(10.0f, places);

		return Round(a * shift) / shift;
	}

	TEST(ActivationFunctions, LinearActivation) 
	{
		EXPECT_EQ(-1.5f, LinearActivation(-1.5f));
		EXPECT_EQ(0.0f, LinearActivation(0.0f));
		EXPECT_EQ(1.5f, LinearActivation(1.5f));
	}

	TEST(ActivationFunctions, BinaryStepActivation) 
	{
		EXPECT_EQ(0.0f, BinaryStepActivation(-1.5f));
		EXPECT_EQ(1.0f, BinaryStepActivation(0.0f));
		EXPECT_EQ(1.0f, BinaryStepActivation(1.5f));
	}

	TEST(ActivationFunctions, SigmoidActivation) 
	{
		EXPECT_NEAR(0.182426f, SigmoidActivation(-1.5f), 0.000001);
		EXPECT_NEAR(0.5f, SigmoidActivation(0.0f), 0.000001);
		EXPECT_NEAR(0.817574f, SigmoidActivation(1.5f), 0.000001);
	}

	TEST(ActivationFunctions, TanhActivation) 
	{
		EXPECT_NEAR(-0.905148f, TanhActivation(-1.5f), 0.000001);
		EXPECT_NEAR(0.0f, TanhActivation(0.0f), 0.000001);
		EXPECT_NEAR(0.905148f, TanhActivation(1.5f), 0.000001);
	}

	TEST(ActivationFunctions, ReLUActivation) 
	{
		EXPECT_EQ(0.0f, ReLUActivation(-1.5));
		EXPECT_EQ(0.0f, ReLUActivation(0.0f));
		EXPECT_EQ(1.5f, ReLUActivation(1.5));
	}

	TEST(ActivationFunctions, LeakyReLUActivation)
	{
		EXPECT_EQ(-0.15f, LeakyReLUActivation(-1.5));
		EXPECT_EQ(0.0f, LeakyReLUActivation(0.0f));
		EXPECT_EQ(1.5f, LeakyReLUActivation(1.5));
	}

	TEST(ActivationFunctions, ParametricReLUActivation) {
		EXPECT_EQ(-0.015f, ParametricReLUActivation(-1.5f, 0.01f));
		EXPECT_EQ(0.0f, ParametricReLUActivation(0.0f, 0.01f));
		EXPECT_EQ(1.5f, ParametricReLUActivation(1.5f, 0.01f));
	}

	TEST(ConnectedLayers, ConnectedLayer)
	{
		std::vector<float> testInputs = { 34.5f };
		std::vector<float> testWeights = { 1.2f, 0.04f, -25.0f };
		std::vector<float> testBiases = { 32.0f };
		int testNodes = 1;
		ACTIVATION testActivation = ACTIVATION::RELU;
		ConnectedLayer* cLayer = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);
		Layer testLayer = cLayer->GetLayer();
		EXPECT_EQ(testInputs, testLayer.inputs);
		EXPECT_EQ(testWeights, testLayer.weights);
		EXPECT_EQ(testBiases, testLayer.biases);
		EXPECT_EQ(testNodes, testLayer.nodes);
		EXPECT_EQ(testActivation, testLayer.activation);
	}

	TEST(ConnectedLayers, SetInputs)
	{
		std::vector<float> testInputs = { 34.5f };
		ConnectedLayer* cLayer = new ConnectedLayer();
		cLayer->SetInputs(testInputs);
		Layer testLayer = cLayer->GetLayer();
		EXPECT_EQ(testInputs, testLayer.inputs);
	}

	TEST(ConnectedLayers, CalculateOutputs)
	{
		std::vector<float> testInputs = { 34.5f, 23.2f, 0.23f };
		std::vector<float> testWeights = { 1.2f, 0.04f, -25.0f };
		std::vector<float> testBiases = { 32.0f };
		int testNodes = 1;
		ACTIVATION testActivation = ACTIVATION::RELU;

		std::vector<float> testOutputs = { 68.578f }; 

		ConnectedLayer* cLayer = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);
		cLayer->CalculateOutputs();
		std::vector<float> layerOutputs = cLayer->GetOutputs();

		EXPECT_EQ(testOutputs, layerOutputs);
	}
}





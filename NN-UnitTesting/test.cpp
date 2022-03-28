#include "pch.h"
#include "../NN-Library/ActivationFunctions.h"
#include "../NN-Library/ActivationFunctions.cpp"
#include "../NN-Library/ConnectedLayer.h"
#include "../NN-Library/ConnectedLayer.cpp"
#include "../NN-Library/NeuralNetwork.h"
#include "../NN-Library/NeuralNetwork.cpp"

namespace NeuralNetworkLibrary
{
	class ActivationsTests : public ::testing::Test
	{
	protected:
		float input1 = -1.5f;
		float input2 = 0.0f;
		float input3 = 1.5f;
	};

	TEST_F(ActivationsTests, LinearActivation)
	{
		EXPECT_EQ(-1.5f, LinearActivation(input1));
		EXPECT_EQ(0.0f, LinearActivation(input2));
		EXPECT_EQ(1.5f, LinearActivation(input3));
	}

	TEST_F(ActivationsTests, BinaryStepActivation)
	{
		EXPECT_EQ(0.0f, BinaryStepActivation(input1));
		EXPECT_EQ(1.0f, BinaryStepActivation(input2));
		EXPECT_EQ(1.0f, BinaryStepActivation(input3));
	}

	TEST_F(ActivationsTests, SigmoidActivation)
	{
		EXPECT_NEAR(0.182426f, SigmoidActivation(input1), 0.000001);
		EXPECT_NEAR(0.5f, SigmoidActivation(input2), 0.000001);
		EXPECT_NEAR(0.817574f, SigmoidActivation(input3), 0.000001);
	}

	TEST_F(ActivationsTests, TanhActivation)
	{
		EXPECT_NEAR(-0.905148f, TanhActivation(input1), 0.000001);
		EXPECT_NEAR(0.0f, TanhActivation(input2), 0.000001);
		EXPECT_NEAR(0.905148f, TanhActivation(input3), 0.000001);
	}

	TEST_F(ActivationsTests, ReLUActivation)
	{
		EXPECT_EQ(0.0f, ReLUActivation(input1));
		EXPECT_EQ(0.0f, ReLUActivation(input2));
		EXPECT_EQ(1.5f, ReLUActivation(input3));
	}

	TEST_F(ActivationsTests, LeakyReLUActivation)
	{
		EXPECT_EQ(-0.15f, LeakyReLUActivation(input1));
		EXPECT_EQ(0.0f, LeakyReLUActivation(input2));
		EXPECT_EQ(1.5f, LeakyReLUActivation(input3));
	}

	TEST_F(ActivationsTests, ParametricReLUActivation) {
		EXPECT_EQ(-0.015f, ParametricReLUActivation(input1, 0.01f));
		EXPECT_EQ(0.0f, ParametricReLUActivation(input2, 0.01f));
		EXPECT_EQ(1.5f, ParametricReLUActivation(input3, 0.01f));
	}

	class ConnectedLayerTest : public ::testing::Test
	{
	protected:
		std::vector<float> testInputs = { 34.5f };
		std::vector<float> testWeights = { 1.2f, 0.04f, -25.0f };
		std::vector<float> testBiases = { 32.0f };
		int testNodes = 1;
		ACTIVATION testActivation = ACTIVATION::RELU;
		ConnectedLayer* cLayer = nullptr;
	};

	TEST_F(ConnectedLayerTest, Constructor)
	{
		cLayer = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);
		EXPECT_EQ(testInputs, cLayer->inputs);
		EXPECT_EQ(testWeights, cLayer->weights);
		EXPECT_EQ(testBiases, cLayer->biases);
		EXPECT_EQ(testNodes, cLayer->nodes);
		EXPECT_EQ(testActivation, cLayer->activation);
	}

	TEST_F(ConnectedLayerTest, SetInput)
	{
		testInputs = { -10.5f };
		cLayer = new ConnectedLayer();
		cLayer->SetInputs(testInputs);
		EXPECT_EQ(testInputs, cLayer->inputs);
	}
	TEST_F(ConnectedLayerTest, SetMultipleInputs)
	{
		testInputs = { 34.5f, 10.5f, -2.0f };
		cLayer = new ConnectedLayer();
		cLayer->SetInputs(testInputs);
		EXPECT_EQ(testInputs, cLayer->inputs);
	}

	TEST_F(ConnectedLayerTest, CalculateOutput)
	{
		testInputs = { 34.5f, 23.2f, 0.23f };
		testWeights = { 1.2f, 0.04f, -25.0f };
		testBiases = { 32.0f };

		std::vector<float> testOutputs = { 68.578f }; 

		cLayer = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);
		cLayer->CalculateOutputs();
		std::vector<float> layerOutputs = cLayer->outputs;

		EXPECT_EQ(testOutputs, layerOutputs);
	}

	TEST_F(ConnectedLayerTest, CalculateMultipleOutputs)
	{
		testInputs = { 34.5f, 23.2f, 0.23f };
		testWeights = { 1.2f, 0.04f, -25.0f, 3.1f, 35.06f, 0.0f, -12.22f, 1.74f, -67.0f };
		testBiases = { 32.0f, 54.0f, -1.0f };
		testNodes = 3;

		std::vector<float> testOutputs = { 68.578f, 974.342f, 0.0f }; 

		cLayer = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);
		cLayer->CalculateOutputs();
		std::vector<float> layerOutputs = cLayer->outputs;

		EXPECT_FLOAT_EQ(testOutputs[0], layerOutputs[0]);
		EXPECT_FLOAT_EQ(testOutputs[1], layerOutputs[1]);
		EXPECT_FLOAT_EQ(testOutputs[2], layerOutputs[2]);
	}

	class NeuralNetworkTest : public ::testing::Test
	{
	protected:
		std::vector<float> testInputs = { 34.5f, 23.2f, 0.23f };
		std::vector<float> testWeights = { 1.2f, 0.04f, -25.0f, 3.1f, 35.06f, 0.0f, -12.22f, 1.74f, -67.0f };
		std::vector<float> testBiases = { 32.0f, 54.0f, -1.0f };
		int testNodes = 3;
		ACTIVATION testActivation = ACTIVATION::RELU;
		ConnectedLayer* cLayer = nullptr;
		ConnectedLayer* cLayer2 = nullptr;
		NeuralNetwork* nn = nullptr;
	};
	TEST_F(NeuralNetworkTest, Constructor)
	{
		nn = new NeuralNetwork();
		EXPECT_EQ(0, nn->GetNumberOfLayers());
	}
	TEST_F(NeuralNetworkTest, AddLayer)
	{
		cLayer = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);
		nn = new NeuralNetwork();
		nn->AddLayer(cLayer);
		std::vector<Layer*> testNN = nn->GetNetwork();

		EXPECT_EQ(testInputs, testNN[0]->inputs);
		EXPECT_EQ(testWeights, testNN[0]->weights);
		EXPECT_EQ(testBiases, testNN[0]->biases);
		EXPECT_EQ(testNodes, testNN[0]->nodes);
		EXPECT_EQ(testActivation, testNN[0]->activation);
		EXPECT_EQ(nn->GetNumberOfLayers(), 1);
	}
	
	TEST_F(NeuralNetworkTest, SetInputsWithNoLayer)
	{
		testInputs = { 34.5f, 23.2f, 0.23f };
		nn = new NeuralNetwork();
		ASSERT_DEATH(nn->SetInputs(testInputs), "There are no layers to set inputs for.");
	}
	TEST_F(NeuralNetworkTest, SetInputs)
	{
		cLayer = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);
		std::vector<float> newInputs = { 23.44f, 0.05f, 102.0f };
		nn = new NeuralNetwork();
		nn->AddLayer(cLayer);
		nn->SetInputs(newInputs);
		std::vector<Layer*> testNN = nn->GetNetwork();
		
		EXPECT_EQ(newInputs, testNN[0]->inputs);
	}

	TEST_F(NeuralNetworkTest, CalculateOutputsWithNoLayer)
	{
		nn = new NeuralNetwork();
		ASSERT_DEATH(nn->CalculateOutputs(), "There are no layers to calculate outputs for.");
	}

	TEST_F(NeuralNetworkTest, CalculateOutputsWithOneLayer)
	{
		cLayer = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);
		std::vector<float> testOutputs = { 68.578f, 974.342f, 0.0f };
		nn = new NeuralNetwork();
		nn->AddLayer(cLayer);
		std::vector<float> outputs = nn->CalculateOutputs();

		EXPECT_FLOAT_EQ(testOutputs[0], outputs[0]);
		EXPECT_FLOAT_EQ(testOutputs[1], outputs[1]);
		EXPECT_FLOAT_EQ(testOutputs[2], outputs[2]);
	}
	
	TEST_F(NeuralNetworkTest, CalculateOutputsWithMultipleLayers)
	{
		cLayer = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);

		testWeights = { 7.0f, 0.4f, -5.0f, -23.1f, 0.06f, 24.0f };
		testBiases = { 12.0f, 4.0f };
		testNodes = 2;
		testActivation = ACTIVATION::RELU;
		cLayer2 = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);

		std::vector<float> testOutputs = { 881.7828f, 0.0f, 0.0f };

		nn = new NeuralNetwork();
		nn->AddLayer(cLayer);
		nn->AddLayer(cLayer2);
		std::vector<float> outputs = nn->CalculateOutputs();

		EXPECT_FLOAT_EQ(testOutputs[0], outputs[0]);
		EXPECT_FLOAT_EQ(testOutputs[1], outputs[1]);
	}

	TEST_F(NeuralNetworkTest, CalculateCost)
	{
		testInputs = { 0.0f, 0.0f };
		testWeights = { 0.2f, 0.8f, 0.4f, 0.6f };
		testBiases = { 0.5f, 0.6f };
		testNodes = 2;
		testActivation = ACTIVATION::SIGMOID;
		cLayer = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);

		testInputs = { 0.0f, 0.0f };
		testWeights = { 0.4f, 0.45f };
		testBiases = { 0.6f };
		testNodes = 1;
		testActivation = ACTIVATION::SIGMOID;
		cLayer2 = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);

		int sets = 4;
		nn = new NeuralNetwork(sets);
		nn->AddLayer(cLayer);
		nn->AddLayer(cLayer2);

		testInputs = { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f };
		std::vector<float> expectedOutputs = { 0.0f, 1.0f, 1.0f, 0.0f };
		nn->TrainNetwork(testInputs, expectedOutputs);
		float cost = nn->GetCost();

		float testCost = 0.323642768f;

		EXPECT_FLOAT_EQ(testCost, cost);
	}
	
	TEST_F(NeuralNetworkTest, BackPropagation)
	{
		testInputs = { 0.0f, 0.0f };
		testWeights = { 0.2f, 0.8f, 0.4f, 0.6f };
		testBiases = { 0.5f, 0.6f };
		testNodes = 2;
		testActivation = ACTIVATION::SIGMOID;
		cLayer = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);

		testInputs = { 0.0f, 0.0f };
		testWeights = { 0.4f, 0.45f };
		testBiases = { 0.6f };
		testNodes = 1;
		testActivation = ACTIVATION::SIGMOID;
		cLayer2 = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);

		int sets = 4;
		nn = new NeuralNetwork(sets);
		nn->AddLayer(cLayer);
		nn->AddLayer(cLayer2);

		testInputs = { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f };
		std::vector<float> expectedOutputs = { 0.0f, 1.0f, 1.0f, 0.0f };

		nn->TrainNetwork(testInputs, expectedOutputs);
		float cost = nn->GetCost();

		for (int iteration = 0; iteration < 10; ++iteration)
		{
			nn->TrainNetwork(testInputs, expectedOutputs);
		}

		float trainedCost = nn->GetCost();

		EXPECT_LT(trainedCost, cost);
		
		for (int iteration = 0; iteration < 1000; ++iteration)
		{
			nn->TrainNetwork(testInputs, expectedOutputs);
		}

		float trainedCost2 = nn->GetCost();

		EXPECT_LT(trainedCost2, trainedCost);
	}

}





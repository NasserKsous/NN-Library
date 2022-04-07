#include "pch.h"
#include "../NN-Library/ActivationFunctions.h"
#include "../NN-Library/ActivationFunctions.cpp"
#include "../NN-Library/ConnectedLayer.h"
#include "../NN-Library/ConnectedLayer.cpp"
#include "../NN-Library/NeuralNetwork.h"
#include "../NN-Library/NeuralNetwork.cpp"
#include "../NN-Library/ConvolutionalLayer.h"
#include "../NN-Library/ConvolutionalLayer.cpp"
#include "../NN-Library/PoolingLayer.h"
#include "../NN-Library/PoolingLayer.cpp"
//Incase of intellisense issues open test options then close and reopen the solution

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
	
	TEST_F(ActivationsTests, SoftmaxActivation)
	{
		std::vector<float> inputs = { 1.0f, 4.2f, 0.6f, 1.23f, 4.3f, 1.2f, 2.5f };
		std::vector<float> outputs = ActivateArray(inputs);
		std::vector<float> expectedOutputs = { 0.01659f, 0.406995f, 0.011121f, 0.020880f, 0.449799f, 0.020263f, 0.07435f };

		for (int i = 0; i < 7; ++i)
		{
			EXPECT_FLOAT_EQ(outputs[i], expectedOutputs[i]);
		}
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

	class ConvolutionalLayerTest : public ::testing::Test
	{
	protected:
		std::vector<float> testInputs = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
										  5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
										  10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
										  15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
										  20.0f, 21.0f, 22.0f, 23.0f, 24.0f };
		std::vector<float> testWeights = { 0.0f, 1.0f, 0.0f,
									  0.0f, 1.0f, 0.0f,
									  0.0f, 1.0f, 0.0f };

		std::vector<Filter> testFilters = { Filter(3, 3, 1, testWeights) };
		
		
		ConvolutionalLayer* convLayer = nullptr;
		ACTIVATION testActivation = ACTIVATION::LINEAR;
	};

	TEST_F(ConvolutionalLayerTest, Constructor)
	{
		convLayer = new ConvolutionalLayer(5, 5, 1, testFilters, 1, 1, false, testActivation);
		std::vector<Filter> filters = convLayer->GetFilters();
		EXPECT_EQ(testFilters[0].values, filters[0].values);
		EXPECT_EQ(testActivation, convLayer->activation);
	}

	TEST_F(ConvolutionalLayerTest, ConstructorMultipleChannels)
	{
		testInputs.clear();
		for (float valueIndex = 0.0f; valueIndex < 75.0f; ++valueIndex)
		{
			testInputs.push_back(valueIndex);
		}
		testWeights = { 0.0f, 1.0f, 0.0f,
						0.0f, 1.0f, 0.0f,
						0.0f, 1.0f, 0.0f,

						0.0f, 1.0f, 0.0f,
						0.0f, 1.0f, 0.0f,
						0.0f, 1.0f, 0.0f,

						0.0f, 1.0f, 0.0f,
						0.0f, 1.0f, 0.0f,
						0.0f, 1.0f, 0.0f };
		testFilters.clear();
		testFilters.push_back(Filter{ 3, 3, 3, testWeights });

		convLayer = new ConvolutionalLayer(5, 5, 3, testFilters, 1, 1, false, testActivation);
		std::vector<Filter> filters = convLayer->GetFilters();
		EXPECT_EQ(testFilters[0].values, filters[0].values);
	}
	
	TEST_F(ConvolutionalLayerTest, ConstructorAssertions)
	{
		testFilters[0].channels = 3;
		ASSERT_DEATH(convLayer = new ConvolutionalLayer(5, 5, 1, testFilters, 1, 1, false, testActivation), "Filter is not the correct size");
	}
	
	TEST_F(ConvolutionalLayerTest, SetInputs)
	{
		convLayer = new ConvolutionalLayer(5, 5, 1, testFilters, 1, 1, false, testActivation);
		
		testInputs = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
					   0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
					   0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
					   0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
					   0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
		convLayer->SetInputs(testInputs);
		EXPECT_EQ(testInputs, convLayer->inputs);

		testInputs = { 0.0f, 1.0f, 2.0f,
					   3.0f, 4.0f, 5.0f,
					   6.0f, 7.0f, 8.0f };
		ASSERT_DEATH(convLayer->SetInputs(testInputs), "Input is not the correct size");
	}

	TEST_F(ConvolutionalLayerTest, SetInputsWithMultipleChannels)
	{
		testInputs.clear();
		for (float valueIndex = 0.0f; valueIndex < 75.0f; ++valueIndex)
		{
			testInputs.push_back(valueIndex);
		}
		testWeights = { 0.0f, 1.0f, 0.0f,
						0.0f, 1.0f, 0.0f,
						0.0f, 1.0f, 0.0f,

						0.0f, 1.0f, 0.0f,
						0.0f, 1.0f, 0.0f,
						0.0f, 1.0f, 0.0f,

						0.0f, 1.0f, 0.0f,
						0.0f, 1.0f, 0.0f,
						0.0f, 1.0f, 0.0f };
		testFilters.clear();
		testFilters.push_back(Filter{ 3, 3, 3, testWeights });

		convLayer = new ConvolutionalLayer(5, 5, 3, testFilters, 1, 1, false, testActivation);
		convLayer->SetInputs(testInputs);
		testInputs.clear();
		for (float valueIndex = 0.0f; valueIndex < 75.0f; ++valueIndex)
		{
			testInputs.push_back(0.0f);
		}
		convLayer->SetInputs(testInputs);
		EXPECT_EQ(testInputs, convLayer->inputs);
	}
	
	TEST_F(ConvolutionalLayerTest, CalculateOutputs)
	{
		convLayer = new ConvolutionalLayer(5, 5, 1, testFilters, 1, 1, false, testActivation);
		convLayer->SetInputs(testInputs);
		convLayer->CalculateOutputs();
		std::vector<float> outputs = convLayer->GetOutputs();
		std::vector<float> expectedOutputs = {18.0f, 21.0f, 24.0f,
											  33.0f, 36.0f, 39.0f,
											  48.0f, 51.0f, 54.0f};

		EXPECT_EQ(expectedOutputs, outputs);
	}
	
	TEST_F(ConvolutionalLayerTest, CalculateOutputsWithPadding)
	{
		convLayer = new ConvolutionalLayer(5, 5, 1, testFilters, 1, 1, true, testActivation);
		convLayer->SetInputs(testInputs);
		convLayer->CalculateOutputs();
		std::vector<float> outputs = convLayer->GetOutputs();
		std::vector<float> expectedOutputs = {5.0f, 7.0f, 9.0f, 11.0f, 13.0f,
											  15.0f, 18.0f, 21.0f, 24.0f, 27.0f,
											  30.0f, 33.0f, 36.0f, 39.0f, 42.0f,
											  45.0f, 48.0f, 51.0f, 54.0f, 57.0f,
											  35.0f, 37.0f, 39.0f, 41.0f, 43.0f};

		EXPECT_EQ(expectedOutputs, outputs);
	}
	
	TEST_F(ConvolutionalLayerTest, CalculateOutputsWithPaddingAndStrideOf2)
	{
		convLayer = new ConvolutionalLayer(5, 5, 1, testFilters, 2, 2, true, testActivation);
		convLayer->SetInputs(testInputs);
		convLayer->CalculateOutputs();
		std::vector<float> outputs = convLayer->GetOutputs();
		std::vector<float> expectedOutputs = {5.0f, 9.0f, 13.0f,
											  30.0f, 36.0f, 42.0f,
											  35.0f, 39.0f, 43.0f};

		EXPECT_EQ(expectedOutputs, outputs);
	}

	TEST_F(ConvolutionalLayerTest, CalculateOutputsWithMultipleChannels)
	{
		testInputs.clear();
		for (float valueIndex = 0.0f; valueIndex < 75.0f; ++valueIndex)
		{
			testInputs.push_back(valueIndex);
		}
		testWeights = { 0.0f, 1.0f, 0.0f,
						0.0f, 1.0f, 0.0f,
						0.0f, 1.0f, 0.0f,

						0.0f, 1.0f, 0.0f,
						0.0f, 1.0f, 0.0f,
						0.0f, 1.0f, 0.0f,

						0.0f, 1.0f, 0.0f,
						0.0f, 1.0f, 0.0f,
						0.0f, 1.0f, 0.0f };
		testFilters.clear();
		testFilters.push_back(Filter{ 3, 3, 3, testWeights });

		convLayer = new ConvolutionalLayer(5, 5, 3, testFilters, 1, 1, false, testActivation);
		convLayer->SetInputs(testInputs);
		convLayer->CalculateOutputs();
		std::vector<float> outputs = convLayer->GetOutputs();
		std::vector<float> expectedOutputs = {  279.0f, 288.0f, 297.0f,
												324.0f, 333.0f, 342.0f,
												369.0f, 378.0f, 387.0f};

		EXPECT_EQ(expectedOutputs, outputs);
	}

	TEST_F(ConvolutionalLayerTest, BackPropagation)
	{
		testWeights = {	0.0f, 1.0f, 
						1.0f, 0.0f};

		testFilters = { Filter(2, 2, 1, testWeights) };

		testInputs = {	0.0f, 1.0f, 2.0f, 
						3.0f, 4.0f, 5.0f,
						6.0f, 7.0f, 8.0f};

		convLayer = new ConvolutionalLayer(3, 3, 1, testFilters, 1, 1, false, ACTIVATION::RELU);
		convLayer->SetInputs(testInputs);
		convLayer->CalculateOutputs();
		std::vector<float> loss = { 0.1f, 0.2f,
									0.2f, 0.1f};
		convLayer->BackPropagate(loss);
		std::vector<float> inputLoss = convLayer->GetInputCosts();
		std::vector<float> weightLoss = convLayer->GetWeightCosts();

		std::vector<float> expectedInputLoss = {0.0f, 0.1f, 0.2f,
												0.1f, 0.4f, 0.1f,
												0.2f, 0.1f, 0.0f};
		
		std::vector<float> expectedWeightLoss = {	1.2f, 1.8f, 
													3.0f, 3.6f};
		for (int i = 0; i < 9; ++i)
		{
			EXPECT_FLOAT_EQ(expectedInputLoss[i], inputLoss[i]);
		}
		for (int i = 0; i < 4; ++i)
		{
			EXPECT_FLOAT_EQ(expectedWeightLoss[i], weightLoss[i]);
		}
	}
	
	TEST_F(ConvolutionalLayerTest, BackPropagationWithMultipleChannels)
	{
		testWeights = {	0.0f, 1.0f, 
						1.0f, 0.0f,
		
						1.0f, 0.0f,
						1.0f, 0.0f,
		
						1.0f, 0.0f,
						0.0f, 1.0f};

		testFilters = { Filter(2, 2, 3, testWeights) };

		testInputs = {	0.0f, 1.0f, 2.0f, 
						3.0f, 4.0f, 5.0f,
						6.0f, 7.0f, 8.0f,
		
						9.0f, 10.0f, 11.0f,
						12.0f, 13.0f, 14.0f,
						15.0f, 16.0f, 17.0f,
		
						18.0f, 19.0f, 20.0f,
						21.0f, 22.0f, 23.0f,
						24.0f, 25.0f, 26.0f};

		convLayer = new ConvolutionalLayer(3, 3, 3, testFilters, 1, 1, false, ACTIVATION::RELU);
		convLayer->SetInputs(testInputs);
		convLayer->CalculateOutputs();
		std::vector<float> loss = { 0.1f, 0.2f,
									0.2f, 0.1f};
		convLayer->BackPropagate(loss);
		std::vector<float> inputLoss = convLayer->GetInputCosts();
		std::vector<float> weightLoss = convLayer->GetWeightCosts();

		std::vector<float> expectedInputLoss = {0.0f, 0.1f, 0.2f,
												0.1f, 0.4f, 0.1f,
												0.2f, 0.1f, 0.0f,
		
												0.1f, 0.2f, 0.0f, 
												0.3f, 0.3f, 0.0f,
												0.2f, 0.1f, 0.0f,
		
												0.1f, 0.2f, 0.0f,
												0.2f, 0.2f, 0.2f,
												0.0f, 0.2f, 0.1f}; /// Sort this out
		
		std::vector<float> expectedWeightLoss = {	1.2f, 1.8f, 
													3.0f, 3.6f,
												
													6.6f, 7.2f,
													8.4f, 9.0f,
		
													12.0f, 12.6f,
													13.8f, 14.4f};
							
		EXPECT_EQ(expectedInputLoss, inputLoss);
		for (int i = 0; i < 12; ++i)
		{
			EXPECT_FLOAT_EQ(expectedWeightLoss[i], weightLoss[i]);
		}
	}
	
	TEST_F(ConvolutionalLayerTest, BackPropagationWithMultipleChannelsAndMultipleFilters)
	{
		testWeights = {	0.0f, 1.0f, 
						1.0f, 0.0f,
		
						1.0f, 0.0f,
						1.0f, 0.0f,
		
						1.0f, 0.0f,
						0.0f, 1.0f};

		std::vector<float> testWeights2 = { 1.0f, 1.0f,
											0.0f, 0.0f,

											0.0f, 1.0f,
											0.0f, 1.0f,

											0.0f, 0.0f,
											1.0f, 1.0f };

		testFilters = { Filter(2, 2, 3, testWeights), Filter(2, 2, 3, testWeights2) };

		testInputs = {	0.0f, 1.0f, 2.0f, 
						3.0f, 4.0f, 5.0f,
						6.0f, 7.0f, 8.0f,
		
						9.0f, 10.0f, 11.0f,
						12.0f, 13.0f, 14.0f,
						15.0f, 16.0f, 17.0f,
		
						18.0f, 19.0f, 20.0f,
						21.0f, 22.0f, 23.0f,
						24.0f, 25.0f, 26.0f};

		convLayer = new ConvolutionalLayer(3, 3, 3, testFilters, 1, 1, false, ACTIVATION::RELU);
		convLayer->SetInputs(testInputs);
		convLayer->CalculateOutputs();
		std::vector<float> loss = { 0.1f, 0.2f,
									0.2f, 0.1f,
		
									0.1f, 0.2f,
									0.3f, 0.4f};
		convLayer->BackPropagate(loss);
		std::vector<float> inputLoss = convLayer->GetInputCosts();
		std::vector<float> weightLoss = convLayer->GetWeightCosts();

		std::vector<float> expectedInputLoss = {0.1f, 0.4f, 0.4f,
												0.4f, 1.1f, 0.5f,
												0.2f, 0.1f, 0.0f,
		
												0.1f, 0.3f, 0.2f, 
												0.3f, 0.7f, 0.6f,
												0.2f, 0.4f, 0.4f,
		
												0.1f, 0.2f, 0.0f,
												0.3f, 0.5f, 0.4f,
												0.3f, 0.9f, 0.5f}; /// Sort this out
		
		std::vector<float> expectedWeightLoss = {	1.2f, 1.8f, 
													3.0f, 3.6f,
												
													6.6f, 7.2f,
													8.4f, 9.0f,
		
													12.0f, 12.6f,
													13.8f, 14.4f};
			
		for (int i = 0; i < 27; ++i)
		{
			EXPECT_FLOAT_EQ(expectedInputLoss[i], inputLoss[i]);
		}
		
		for (int i = 0; i < 12; ++i)
		{
			EXPECT_FLOAT_EQ(expectedWeightLoss[i], weightLoss[i]);
		}
	}

	class PoolingLayerTest : public ::testing::Test
	{
	protected:
		std::vector<float> testInputs = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
										  5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
										  10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
										  15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
										  20.0f, 21.0f, 22.0f, 23.0f, 24.0f };

		PoolingLayer* poolLayer = nullptr;
	};

	TEST_F(PoolingLayerTest, Constructor)
	{
		poolLayer = new PoolingLayer(5, 5, 1, testInputs, 3, 3, 1, 1, true);
		EXPECT_EQ(testInputs, poolLayer->inputs);
	}

	TEST_F(PoolingLayerTest, ConstructorMultipleChannels)
	{
		testInputs.clear();
		for (float valueIndex = 0.0f; valueIndex < 75.0f; ++valueIndex)
		{
			testInputs.push_back(valueIndex);
		}

		poolLayer = new PoolingLayer(5, 5, 3, testInputs, 3, 3, 1, 1, true);
	}

	TEST_F(PoolingLayerTest, ConstructorAssertions)
	{
		ASSERT_DEATH(poolLayer = new PoolingLayer(3, 3, 3, testInputs, 3, 3, 1, 1, true), "Input is not the correct size");
	}

	TEST_F(PoolingLayerTest, SetInputs)
	{
		poolLayer = new PoolingLayer(5, 5, 1, testInputs, 3, 3, 1, 1, true);

		testInputs = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
					   0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
					   0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
					   0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
					   0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
		poolLayer->SetInputs(testInputs);
		EXPECT_EQ(testInputs, poolLayer->inputs);

		poolLayer = new PoolingLayer(5, 5, 1, testInputs, 3, 3, 1, 1, true);
		testInputs = { 0.0f, 1.0f, 2.0f,
					   3.0f, 4.0f, 5.0f,
					   6.0f, 7.0f, 8.0f };
		ASSERT_DEATH(poolLayer->SetInputs(testInputs), "Input is not the correct size");
	}

	TEST_F(PoolingLayerTest, SetInputsWithMultipleChannels)
	{
		testInputs.clear();
		for (float valueIndex = 0.0f; valueIndex < 75.0f; ++valueIndex)
		{
			testInputs.push_back(valueIndex);
		}
		poolLayer = new PoolingLayer(5, 5, 3, testInputs, 3, 3, 1, 1, true);

		testInputs.clear();
		for (float valueIndex = 0.0f; valueIndex < 75.0f; ++valueIndex)
		{
			testInputs.push_back(0.0f);
		}
		poolLayer->SetInputs(testInputs);
		EXPECT_EQ(testInputs, poolLayer->inputs);
	}

	TEST_F(PoolingLayerTest, CalculateOutputsMaxPooling)
	{
		poolLayer = new PoolingLayer(5, 5, 1, testInputs, 3, 3, 1, 1, true);
		poolLayer->CalculateOutputs();
		std::vector<float> outputs = poolLayer->GetOutputs();
		std::vector<float> expectedOutputs = { 12.0f, 13.0f, 14.0f,
											  17.0f, 18.0f, 19.0f,
											  22.0f, 23.0f, 24.0f };

		EXPECT_EQ(expectedOutputs, outputs);
	}
	
	TEST_F(PoolingLayerTest, CalculateOutputsAveragePooling)
	{
		poolLayer = new PoolingLayer(5, 5, 1, testInputs, 3, 3, 1, 1, false);
		poolLayer->CalculateOutputs();
		std::vector<float> outputs = poolLayer->GetOutputs();
		std::vector<float> expectedOutputs = { 6.0f, 7.0f, 8.0f,
											  11.0f, 12.0f, 13.0f,
											  16.0f, 17.0f, 18.0f };

		EXPECT_EQ(expectedOutputs, outputs);
	}
	
	TEST_F(PoolingLayerTest, CalculateOutputsMaxPoolingAndStrideOf2)
	{
		testInputs = {	0.0f, 1.0f, 2.0f, 3.0f,
						4.0f, 5.0f, 6.0f, 7.0f,
						8.0f, 9.0f, 10.0f, 11.0f,
						12.0f, 13.0f, 14.0f, 15.0f };
		poolLayer = new PoolingLayer(4, 4, 1, testInputs, 2, 2, 2, 2, true);
		poolLayer->CalculateOutputs();
		std::vector<float> outputs = poolLayer->GetOutputs();
		std::vector<float> expectedOutputs = {	5.0f, 7.0f, 
												13.0f, 15.0f};

		EXPECT_EQ(expectedOutputs, outputs);
	}
	
	TEST_F(PoolingLayerTest, CalculateOutputsAveragePoolingAndStrideOf2)
	{
		testInputs = { 0.0f, 1.0f, 2.0f, 3.0f,
						4.0f, 5.0f, 6.0f, 7.0f,
						8.0f, 9.0f, 10.0f, 11.0f,
						12.0f, 13.0f, 14.0f, 15.0f };
		poolLayer = new PoolingLayer(4, 4, 1, testInputs, 2, 2, 2, 2, false);
		poolLayer->CalculateOutputs();
		std::vector<float> outputs = poolLayer->GetOutputs();
		std::vector<float> expectedOutputs = { 2.5f, 4.5f,
												10.5f, 12.5f };

		EXPECT_EQ(expectedOutputs, outputs);
	}

	TEST_F(PoolingLayerTest, CalculateOutputsMaxPoolingWithMultipleChannels)
	{
		testInputs.clear();
		for (float valueIndex = 0.0f; valueIndex < 75.0f; ++valueIndex)
		{
			testInputs.push_back(valueIndex);
		}

		poolLayer = new PoolingLayer(5, 5, 3, testInputs, 3, 3, 1, 1, true);
		poolLayer->CalculateOutputs();
		std::vector<float> outputs = poolLayer->GetOutputs();
		std::vector<float> expectedOutputs = {	12.0f, 13.0f, 14.0f,
												17.0f, 18.0f, 19.0f,
												22.0f, 23.0f, 24.0f,
			
												37.0f, 38.0f, 39.0f,
												42.0f, 43.0f, 44.0f,
												47.0f, 48.0f, 49.0f,
												
												62.0f, 63.0f, 64.0f,
												67.0f, 68.0f, 69.0f,
												72.0f, 73.0f, 74.0f
											};

		EXPECT_EQ(expectedOutputs, outputs);
	}
	
	TEST_F(PoolingLayerTest, CalculateOutputsAveragePoolingWithMultipleChannels)
	{
		testInputs.clear();
		for (float valueIndex = 0.0f; valueIndex < 75.0f; ++valueIndex)
		{
			testInputs.push_back(valueIndex);
		}

		poolLayer = new PoolingLayer(5, 5, 3, testInputs, 3, 3, 1, 1, false);
		poolLayer->CalculateOutputs();
		std::vector<float> outputs = poolLayer->GetOutputs();
		std::vector<float> expectedOutputs = {	6.0f, 7.0f, 8.0f,
												11.0f, 12.0f, 13.0f,
												16.0f, 17.0f, 18.0f,
			
												31.0f, 32.0f, 33.0f,
												36.0f, 37.0f, 38.0f,
												41.0f, 42.0f, 43.0f,
												
												56.0f, 57.0f, 58.0f,
												61.0f, 62.0f, 63.0f,
												66.0f, 67.0f, 68.0f
											};

		EXPECT_EQ(expectedOutputs, outputs);
	}

	TEST_F(PoolingLayerTest, BackPropagation)
	{
		testInputs = {	0.0f, 1.0f, 2.0f, 3.0f,
						4.0f, 5.0f, 6.0f, 7.0f, 
						8.0f, 9.0f, 10.0f, 11.0f, 
						12.0f, 13.0f, 14.0f, 15.0f,  };
		poolLayer = new PoolingLayer(4, 4, 1, testInputs, 2, 2, 2, 2, true);
		poolLayer->CalculateOutputs();

		std::vector<float> inputLoss = {	1.0f, 2.0f, 
											3.0f, 4.0f };

		poolLayer->BackPropagate(inputLoss);

		std::vector<float> lossInput = poolLayer->GetInputCosts();

		std::vector<float> expectedLoss = { 0.0f, 0.0f, 0.0f, 0.0f,
											0.0f, 1.0f, 0.0f, 2.0f,
											0.0f, 0.0f, 0.0f, 0.0f,
											0.0f, 3.0f, 0.0f, 4.0f };

		EXPECT_EQ(lossInput, expectedLoss);
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

		ConvolutionalLayer* convLayer = nullptr;
		PoolingLayer* poolLayer = nullptr;

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

	TEST_F(NeuralNetworkTest, CalculateOutputsWithMultipleDifferentLayers)
	{
		testInputs = {	0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
						5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
						10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
						15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
						20.0f, 21.0f, 22.0f, 23.0f, 24.0f };

		std::vector<float> testValues = { 0.0f, 1.0f, 0.0f,
									  0.0f, 1.0f, 0.0f,
									  0.0f, 1.0f, 0.0f };

		std::vector<Filter> testFilters = { Filter(3, 3, 1, testValues) };

		convLayer = new ConvolutionalLayer(5, 5, 1, testFilters, 1, 1, true, testActivation);

		poolLayer = new PoolingLayer(5, 5, 1, testInputs, 3, 3, 1, 1, true);

		testNodes = 1;

		testBiases = { 5.0f };

		cLayer = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, ACTIVATION::LINEAR);

		

		std::vector<float> testOutputs = { -3297.16f };

		nn = new NeuralNetwork();
		nn->AddLayer(convLayer);
		nn->AddLayer(poolLayer);
		nn->AddLayer(cLayer);
		nn->SetInputs(testInputs);
		std::vector<float> outputs = nn->CalculateOutputs();

		EXPECT_FLOAT_EQ(testOutputs[0], outputs[0]);
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





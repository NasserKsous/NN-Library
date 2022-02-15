#include "pch.h"
#include "../NN-Library/ActivationFunctions.h"

static float Round(float a)
{
	return (a > 0) ? ::floor(a + 0.5f) : ::ceil(a - 0.5f);
}
static float Round(float a, int places)
{
	const float shift = pow(10.0f, places);

	return Round(a * shift) / shift;
}

TEST(ActivationFunctions, LinearActivation) {
	EXPECT_EQ(-1.5f, LinearActivation(-1.5f));
	EXPECT_EQ(0.0f, LinearActivation(0.0f));
	EXPECT_EQ(1.5f, LinearActivation(1.5f));
}

TEST(ActivationFunctions, BinaryStepActivation) {
	EXPECT_EQ(0.0f, BinaryStepActivation(-1.5f));
	EXPECT_EQ(1.0f, BinaryStepActivation(0.0f));
	EXPECT_EQ(1.0f, BinaryStepActivation(1.5f));
}

TEST(ActivationFunctions, SigmoidActivation) {
	EXPECT_EQ(Round(0.182426f, 6), Round(SigmoidActivation(-1.5f), 6));
	EXPECT_EQ(Round(0.5f, 6), Round(SigmoidActivation(0.0f), 6));
	EXPECT_EQ(Round(0.817574f, 6), Round(SigmoidActivation(1.5f), 6));
}

TEST(ActivationFunctions, TanhActivation) {
	EXPECT_EQ(Round(-0.905148f,6), Round(TanhActivation(-1.5f), 6));
	EXPECT_EQ(Round(0.0f,6), Round(TanhActivation(0.0f), 6));
	EXPECT_EQ(Round(0.905148f, 6), Round(TanhActivation(1.5f), 6));
}

TEST(ActivationFunctions, ReLUActivation) {
	EXPECT_EQ(0.0f, ReLUActivation(-1.5));
	EXPECT_EQ(0.0f, ReLUActivation(0.0f));
	EXPECT_EQ(1.5f, ReLUActivation(1.5));
}

TEST(ActivationFunctions, LeakyReLUActivation) {
	EXPECT_EQ(-0.15f, LeakyReLUActivation(-1.5));
	EXPECT_EQ(0.0f, LeakyReLUActivation(0.0f));
	EXPECT_EQ(1.5f, LeakyReLUActivation(1.5));
}

TEST(ActivationFunctions, ParametricReLUActivation) {
	EXPECT_EQ(-0.015f, ParametricReLUActivation(-1.5f, 0.01f));
	EXPECT_EQ(0.0f, ParametricReLUActivation(0.0f, 0.01f));
	EXPECT_EQ(1.5f, ParametricReLUActivation(1.5f, 0.01f));
}



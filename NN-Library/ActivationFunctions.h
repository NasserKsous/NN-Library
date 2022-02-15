#pragma once
#define _USE_MATH_DEFINES
#include "math.h"
#include "Constants.h"

static inline float LinearActivation(float x) { return x; }
static inline float BinaryStepActivation(float x) { return (x >= 0) ? 1.0f : 0.0f; }
static inline float SigmoidActivation(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float TanhActivation(float x) { return (expf(x) - expf(-x))/(expf(x) + expf(-x)); }
static inline float ReLUActivation(float x) { return (x >= 0) ? x : 0.0f; }
static inline float LeakyReLUActivation(float x) { return (x >= 0) ? x : 0.1f*x; }
static inline float ParametricReLUActivation(float x, float alpha) { return (x >= 0) ? x : alpha * x; }

class ActivationFunction
{
	ActivationFunction(ACTIVATION type);
	ActivationFunction(ACTIVATION type, float a);
	ACTIVATION activationType;
	float alpha;
	float Activate(float x);
};
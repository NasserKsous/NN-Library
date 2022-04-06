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

static inline float LinearDeactivation(float x) { return x; }
static inline float BinaryStepDeactivation(float x) { return (x); }
static inline float SigmoidDeactivation(float x) { return  x * (1.0f - x); }
static inline float TanhDeactivation(float x) { return (1 - (x * x)); }
static inline float ReLUDeactivation(float x) { return (x >= 0) ? x : 0.0f; }
static inline float LeakyReLUDeactivation(float x) { return (x >= 0) ? x : 10.0f*x; }

float Activate(float x, ACTIVATION type);
float Deactivate(float x, ACTIVATION type);

//class ActivationFunction
//{
//public:
//	ActivationFunction(ACTIVATION type);
//	ActivationFunction(ACTIVATION type, float a);
//
//	float Activate(float x, ACTIVATION type);
//
//private:
//	ACTIVATION activationType;
//	float alpha;
//};
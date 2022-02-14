#pragma once
#define _USE_MATH_DEFINES
#include "math.h"

static inline float LinearActivation(float x) { return x; }
static inline float BinaryStepActivation(float x) { return (x >= 0) ? 1 : 0; }
static inline float SigmoidActivation(float x) { return 1 / (1 + expf(-x)); }
static inline float TanhActivation(float x) { return (expf(x) - expf(x))/(expf(x) + expf(-x)); }
static inline float ReLUActivation(float x) { return (x >= 0) ? x : 0; }
static inline float LeakyReLUActivation(float x) { return (x >= 0) ? x : 0.1*x; }
static inline float ParametricReLUActivation(float x, float alpha) { return (x >= 0) ? x : alpha * x; }
static inline float GeLUActivation(float x) { return (0.5 * x) * (1 + tanhf(sqrt(2/M_PI)) * (x + 0.044715 * powf(x, 3))); }
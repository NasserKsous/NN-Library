#include "ActivationFunctions.h"

ActivationFunction::ActivationFunction(ACTIVATION type)
{
	activationType = type;
}

ActivationFunction::ActivationFunction(ACTIVATION type, float a)
{
	activationType = type;
	alpha = a;
}

float ActivationFunction::Activate(float x)
{
	switch (activationType)
	{
		case LINEAR:
		{
			return LinearActivation(x);
			break;
		}
		case BINARY_STEP:
		{
			return BinaryStepActivation(x);
			break;
		}
		case SIGMOID:
		{
			return SigmoidActivation(x);
			break;
		}
		case TANH:
		{
			return TanhActivation(x);
			break;
		}
		case RELU:
		{
			return ReLUActivation(x);
			break;
		}
		case LEAKY_RELU:
		{
			return LeakyReLUActivation(x);
			break;
		}
		case PARAMETRIC_RELU:
		{
			return ParametricReLUActivation(x, alpha);
			break;
		}

		
	}
	return 0.0f;
}
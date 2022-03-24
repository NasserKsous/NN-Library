#include "ActivationFunctions.h"

ActivationFunction::ActivationFunction(ACTIVATION type)
{
	activationType = type;
	alpha = 0.0f;
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
	case ACTIVATION::LINEAR:
		{
			return LinearActivation(x);
			break;
		}
		case ACTIVATION::BINARY_STEP:
		{
			return BinaryStepActivation(x);
			break;
		}
		case ACTIVATION::SIGMOID:
		{
			return SigmoidActivation(x);
			break;
		}
		case ACTIVATION::TANH:
		{
			if (x < -1.0f)
				return -1.0f;
			if (x > 1.0f)
				return 1.0f;
			return TanhActivation(x);
			break;
		}
		case ACTIVATION::RELU:
		{
			return ReLUActivation(x);
			break;
		}
		case ACTIVATION::LEAKY_RELU:
		{
			return LeakyReLUActivation(x);
			break;
		}
		case ACTIVATION::PARAMETRIC_RELU:
		{
			return ParametricReLUActivation(x, alpha);
			break;
		}

		
	}
	return 0.0f;
}
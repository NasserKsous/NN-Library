#include "ActivationFunctions.h"

float Activate(float x, ACTIVATION type)
{
	switch (type)
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
	}
	return 0.0f;
}

float Deactivate(float x, ACTIVATION type)
{
	switch (type)
	{
	case ACTIVATION::LINEAR:
	{
		return LinearDeactivation(x);
		break;
	}
	case ACTIVATION::BINARY_STEP:
	{
		return BinaryStepDeactivation(x);
		break;
	}
	case ACTIVATION::SIGMOID:
	{
		return SigmoidDeactivation(x);
		break;
	}
	case ACTIVATION::TANH:
	{
		if (x < -1.0f)
			return -1.0f;
		if (x > 1.0f)
			return 1.0f;
		return TanhDeactivation(x);
		break;
	}
	case ACTIVATION::RELU:
	{
		return ReLUDeactivation(x);
		break;
	}
	case ACTIVATION::LEAKY_RELU:
	{
		return LeakyReLUDeactivation(x);
		break;
	}
	}
	return 0.0f;
}

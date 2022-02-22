// NN-Library.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "ActivationFunctions.h"
#include "ConnectedLayer.h"

int main()
{
    float x = 1.5f;
    float y = 0.0f;

    y = LinearActivation(x);
    std::cout << "Linear Activation - \nx: " << x << "\ny: " << y << "\n\n";

    y = BinaryStepActivation(x);
    std::cout << "Binary Step Activation - \nx: " << x << "\ny: " << y << "\n\n";

    y = SigmoidActivation(x);
    std::cout << "Sigmoid Activation - \nx: " << x << "\ny: " << y << "\n\n";

    y = TanhActivation(x);
    std::cout << "Tanh Activation - \nx: " << x << "\ny: " << y << "\n\n";

    y = ReLUActivation(x);
    std::cout << "ReLU Activation - \nx: " << x << "\ny: " << y << "\n\n";

    y = LeakyReLUActivation(x);
    std::cout << "Leaky ReLU Activation - \nx: " << x << "\ny: " << y << "\n\n";

    y = ParametricReLUActivation(x, 0.1f);
    std::cout << "Parametric ReLU Activation - \nx: " << x << "\ny: " << y << "\n\n";

    std::vector<float> testInputs = { 34.5f };
    std::vector<float> testWeights = { 1.2f, 0.04f, -25.0f };
    std::vector<float> testBiases = { 32.0f };
    int testNodes = 3;
    ACTIVATION testActivation = ACTIVATION::RELU;
    ConnectedLayer* cLayer = new ConnectedLayer(testInputs, testWeights, testBiases, testNodes, testActivation);


}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file

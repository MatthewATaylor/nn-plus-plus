#pragma once

#include <iostream>
#include <random>

#include "activation/LinearActivation.h"
#include "loss/MeanSquaredErrorLoss.h"
#include "optimizer/AdamOptimizer.h"
#include "Dense.h"
#include "Network.h"
#include "math/Vec.h"

int main() {
	const size_t DATA_SIZE = 500;
	const size_t INPUT_SIZE = 2;
	const size_t OUTPUT_SIZE = 1;

	Vec<float> inputs[DATA_SIZE];
	Vec<float> targets[DATA_SIZE];

	for (size_t i = 0; i < DATA_SIZE; ++i) {
		std::random_device randomDevice;
		std::mt19937 generator(randomDevice());
		std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

		float num1 = distribution(generator);
		float num2 = distribution(generator);
		float sum = num1 + num2;

		inputs[i] = { num1, num2 };
		targets[i] = { sum };
	}

	LinearActivation linear;

	Dense dense(INPUT_SIZE, OUTPUT_SIZE, &linear);
	Network network{ &dense };

	MeanSquaredErrorLoss loss;
	AdamOptimizer optimizer;
	network.train(inputs, targets, DATA_SIZE, &loss, &optimizer, 1000);

	std::cout << "\n";
	std::cout << "1 + 1 = " << network.evaluate({ { 1.0f }, { 1.0f } })(0, 0) << "\n";
	std::cout << "25 + 6 = " << network.evaluate({ { 25.0f }, { 6.0f } })(0, 0) << "\n";
	std::cout << "0.2 + 0.12 = " << network.evaluate({ { 0.2f }, { 0.12f } })(0, 0) << "\n";

	std::cin.get();
	return 0;
}

#pragma once

#include <iostream>
#include <random>

#include "activation/LinearActivation.h"
#include "loss/MeanSquaredErrorLoss.h"
#include "optimizer/MomentumOptimizer.h"
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
	MomentumOptimizer optimizer(0.001f, 0.9f);
	network.train(inputs, targets, DATA_SIZE, &loss, &optimizer, 100);

	std::cout << network.evaluate({ 0.1f, 0.3f }) << "\n";
	std::cout << network.evaluate({ 0.4f, 0.22f }) << "\n";
	std::cout << network.evaluate({ 0.25f, 0.7f }) << "\n";
	std::cout << network.evaluate({ 0.0f, 0.0f }) << "\n";
	std::cout << network.evaluate({ 5.0f, 12.5f }) << "\n";
	std::cout << network.evaluate({ 10.6f, 9.8f }) << "\n";

	std::cin.get();
	return 0;
}

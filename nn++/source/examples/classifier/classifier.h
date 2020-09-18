#pragma once

#include <iostream>
#include <random>

#include "activation/ReLUActivation.h"
#include "activation/SoftmaxActivation.h"
#include "loss/CategoricalCrossEntropyLoss.h"
#include "optimizer/MomentumOptimizer.h"
#include "Dense.h"
#include "Network.h"
#include "math/Vec.h"

int main() {
	const size_t DATA_SIZE = 500;
	const size_t INPUT_SIZE = 1;
	const size_t OUTPUT_SIZE = 4;
	const size_t HIDDEN_UNITS = 4;

	Vec<float> inputs[DATA_SIZE];
	Vec<float> targets[DATA_SIZE];

	//_control87(_EM_INEXACT | _EM_UNDERFLOW | _EM_DENORMAL, _MCW_EM);

	for (size_t i = 0; i < DATA_SIZE; ++i) {
		std::random_device randomDevice;
		std::mt19937 generator(randomDevice());
		std::uniform_real_distribution<double> distribution(0.0, 1.0);

		float input = (float) distribution(generator);

		Vec<float> target(OUTPUT_SIZE);
		if (input < 0.25f) {
			target = { 1.0f, 0.0f, 0.0f, 0.0f };
		}
		else if (input < 0.5f) {
			target = { 0.0f, 1.0f, 0.0f, 0.0f };
		}
		else if (input < 0.75f) {
			target = { 0.0f, 0.0f, 1.0f, 0.0f };
		}
		else {
			target = { 0.0f, 0.0f, 0.0f, 1.0f };
		}

		inputs[i] = { input };
		targets[i] = target;
	}

	ReLUActivation relu;
	SoftmaxActivation softmax;

	Dense dense1(INPUT_SIZE, HIDDEN_UNITS, &relu);
	Dense dense2(HIDDEN_UNITS, HIDDEN_UNITS, &relu);
	Dense dense3(HIDDEN_UNITS, OUTPUT_SIZE, &softmax);

	Network network{
		&dense1,
		&dense2,
		&dense3
	};

	CategoricalCrossEntropyLoss loss;
	MomentumOptimizer optimizer(0.0075f, 0.9f);
	network.train(inputs, targets, DATA_SIZE, &loss, &optimizer, 5000);

	std::cout << network.evaluate({ 0.1f }) << "\n";
	std::cout << network.evaluate({ 0.2f }) << "\n";
	std::cout << network.evaluate({ 0.3f }) << "\n";
	std::cout << network.evaluate({ 0.4f }) << "\n";
	std::cout << network.evaluate({ 0.5f }) << "\n";
	std::cout << network.evaluate({ 0.6f }) << "\n";
	std::cout << network.evaluate({ 0.7f }) << "\n";
	std::cout << network.evaluate({ 0.8f }) << "\n";
	std::cout << network.evaluate({ 0.9f }) << "\n";

	std::cin.get();
	return 0;
}

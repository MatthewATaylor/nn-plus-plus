#pragma once

#include <array>
#include <iostream>
#include <random>

#include "activation/ReLUActivation.h"
#include "activation/LinearActivation.h"
#include "activation/SoftmaxActivation.h"
#include "loss/MeanSquaredErrorLoss.h"
#include "Dense.h"
#include "Network.h"
#include "math/Vec.h"

int main() {
	const size_t DATA_SIZE = 500;
	const size_t INPUT_SIZE = 2;
	const size_t OUTPUT_SIZE = 1;
	const size_t HIDDEN_UNITS = 32;
	const size_t BATCH_SIZE = 10;

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

	ReLUActivation relu;
	LinearActivation linear;

	//Dense dense1(INPUT_SIZE, HIDDEN_UNITS, &relu);
	//Dense dense2(HIDDEN_UNITS, HIDDEN_UNITS, &relu);
	//Dense dense3(HIDDEN_UNITS, OUTPUT_SIZE, &linear);

	//Network network {
	//	&dense1,
	//	&dense2,
	//	&dense3
	//};

	Dense dense1(INPUT_SIZE, OUTPUT_SIZE, &linear);
	//Dense dense2(HIDDEN_UNITS, OUTPUT_SIZE, &linear);
	Network network{
		&dense1,
		//&dense2
	};

	MeanSquaredErrorLoss loss;
	network.train(inputs, targets, DATA_SIZE, &loss, 0.0000000001f, 10000);

	std::cin.get();
	return 0;
}

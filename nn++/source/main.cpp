#include <array>
#include <iostream>
#include <random>

#include "Activation.h"
#include "Loss.h"
#include "Dense.h"
#include "Network.h"
#include "math/Vec.h"

int main() {
	const size_t NUM_INPUTS = 512;
	const size_t INPUT_SIZE = 2;
	const size_t OUTPUT_SIZE = 1;
	const size_t HIDDEN_UNITS = 16;
	const size_t BATCH_SIZE = 10;
	
	std::array<Vec<float, INPUT_SIZE>, NUM_INPUTS> x;
	std::array<Vec<float, OUTPUT_SIZE>, NUM_INPUTS> y;
	
	for (size_t i = 0; i < NUM_INPUTS; ++i) {
		std::random_device randomDevice;
		std::mt19937 generator(randomDevice());
		std::uniform_real_distribution<float> distribution(-10000.0f, 10000.0f);
		
		float num1 = distribution(generator);
		float num2 = distribution(generator);
		float sum = num1 + num2;

		x[i].set(0, num1);
		x[i].set(1, num2);
		y[i].set(0, sum);
	}

	Dense<INPUT_SIZE, HIDDEN_UNITS, Activation::ReLU> dense1;
	Dense<HIDDEN_UNITS, HIDDEN_UNITS, Activation::ReLU> dense2;
	Dense<HIDDEN_UNITS, OUTPUT_SIZE, Activation::Linear> dense3;
	Network network(
		&dense1,
		&dense2,
		&dense3
	);

	Vec<float, INPUT_SIZE> input(1.0f, 2.0f);
	dense1.evaluate(input);
	dense2.evaluate(dense1.activations);
	dense3.evaluate(dense2.activations);
	//std::cout << dense3.activations << "\n";

	Vec<float, 1> target(3.0f);
	//std::cout << 
	//	network.getLoss<INPUT_SIZE, OUTPUT_SIZE, Loss::MeanSquaredError>(
	//		input, target
	//	) << "\n";

	network.train<NUM_INPUTS, INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE, Loss::MeanSquaredError>(
		x, y
	);

	std::cin.get();
	return 0;
}

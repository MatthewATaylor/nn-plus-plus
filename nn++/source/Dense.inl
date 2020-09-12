#pragma once

inline Dense::Dense(size_t inputSize, size_t units, const Activation *activation) :
	activation(activation), inputSize(inputSize), units(units),
	weights(units, inputSize), biases(units, 0), weightedInputs(units),
	activations(units), errors(units) {

	//Randomly initialize weights
	std::random_device randomDevice;
	std::mt19937 generator(randomDevice());
	std::normal_distribution<float> distribution(0.0f, 1.0f);
	for (size_t i = 0; i < units; ++i) {
		for (size_t j = 0; j < inputSize; ++j) {
			weights(i, j) = (float) distribution(generator) * 0.01f;
		}
	}
}

inline void Dense::evaluate(const Vec<float> &input) {
	weightedInputs = weights * input + biases;
	activations = activation->func(weightedInputs);
}

inline Vec<float> Dense::activationFuncDerivative() const {
	return activation->derivative(weightedInputs);
}

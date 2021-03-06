#pragma once

inline Dense::Dense(size_t inputSize, size_t units, const Activation *activation) :
	activation(activation), inputSize(inputSize), units(units),
	weights(units, inputSize), biases(units, 0.0f),
	weightV(units, inputSize, 0.0f), biasV(units, 0.0f),
	weightS(units, inputSize, 0.0f), biasS(units, 0.0f) {

	//Randomly initialize weights
	std::random_device randomDevice;
	std::mt19937 generator(randomDevice());
	std::normal_distribution<float> distribution(0.0f, 1.0f);
	float weightFactor = std::sqrt(activation->getWeightFactor() / (float) units);
	for (size_t i = 0; i < units; ++i) {
		for (size_t j = 0; j < inputSize; ++j) {
			weights(i, j) = (float) distribution(generator) * weightFactor;
		}
	}
}

inline void Dense::evaluate(const Mat<float> &input) {
	weightedInputs = weights * input;

	//Add biases to each weighted input in batch
	for (size_t i = 0; i < weightedInputs.rows; ++i) {
		for (size_t j = 0; j < weightedInputs.cols; ++j) {
			weightedInputs(i, j) += biases(i);
		}
	}

	activations = activation->func(weightedInputs);
}

inline Mat<float> Dense::activationFuncDerivative() const {
	return activation->derivative(weightedInputs);
}

inline const Activation *Dense::getActivation() const {
	return activation;
}

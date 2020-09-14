#pragma once

inline Vec<float> SoftmaxActivation::func(const Vec<float> &input) const {
	float exponentialSum = 0.0f;
	for (size_t i = 0; i < input.size; ++i) {
		exponentialSum += std::exp(input(i));
	}

	Vec<float> activation(input.size);
	for (size_t i = 0; i < input.size; ++i) {
		activation(i) = std::exp(input(i)) / exponentialSum;
	}

	return activation;
}

inline Vec<float> SoftmaxActivation::derivative(const Vec<float> &input) const {
	throw std::exception(
		"Softmax may only be used in an output layer and with categorical cross-entropy loss"
	);
}

#pragma once

inline ReLUActivation::ReLUActivation() {
	weightFactor = 2.0f;
}

inline Vec<float> ReLUActivation::func(const Vec<float> &input) const {
	Vec<float> activation(input.size);
	for (size_t i = 0; i < input.size; ++i) {
		float inputValue = input(i);
		float newValue = inputValue < 0.0f ? 0.0f : inputValue;
		activation(i) = newValue;
	}
	return activation;
}

inline Vec<float> ReLUActivation::derivative(const Vec<float> &input) const {
	Vec<float> derivatives(input.size);
	for (size_t i = 0; i < input.size; ++i) {
		float inputValue = input(i);
		float newValue = inputValue < 0.0f ? 0.0f : 1.0f;
		derivatives(i) = newValue;
	}
	return derivatives;
}

#pragma once

inline Vec<float> LeakyReLUActivation::func(const Vec<float> &input) const {
	Vec<float> activation(input.size);
	for (size_t i = 0; i < input.size; ++i) {
		float inputValue = input(i);
		float newValue = inputValue < 0.0f ? inputValue * 0.01f : inputValue;
		activation(i) = newValue;
	}
	return activation;
}

inline Vec<float> LeakyReLUActivation::derivative(const Vec<float> &input) const {
	Vec<float> derivatives(input.size);
	for (size_t i = 0; i < input.size; ++i) {
		float inputValue = input(i);
		float newValue = inputValue < 0.0f ? 0.01f : 1;
		derivatives(i) = newValue;
	}
	return derivatives;
}

#pragma once

inline Vec<float> TanhActivation::func(const Vec<float> &input) const {
	Vec<float> activation(input.size);
	for (size_t i = 0; i < input.size; ++i) {
		float inputValue = input(i);
		float newValue = 2.0f / (1.0f + std::exp(-2.0f * inputValue)) - 1.0f;
		activation(i) = newValue;
	}
	return activation;
}

inline Vec<float> TanhActivation::derivative(const Vec<float> &input) const {
	Vec<float> derivatives(input.size);
	for (size_t i = 0; i < input.size; ++i) {
		float inputValue = input(i);
		float exponential = std::exp(-2.0f * inputValue);
		float newValue = 4.0f * exponential /
			((1.0f + exponential) * (1.0f + exponential));
		derivatives(i) = newValue;
	}
	return derivatives;
}

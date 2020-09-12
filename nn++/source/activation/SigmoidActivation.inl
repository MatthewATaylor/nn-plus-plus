#pragma once

inline Vec<float> SigmoidActivation::func(const Vec<float> &input) const {
	Vec<float> activation(input.size);
	for (size_t i = 0; i < input.size; ++i) {
		float inputValue = input(i);
		float newValue = 1.0f / (1.0f + std::exp(-inputValue));
		activation(i) = newValue;
	}
	return activation;
}

inline Vec<float> SigmoidActivation::derivative(const Vec<float> &input) const {
	Vec<float> derivatives(input.size);
	for (size_t i = 0; i < input.size; ++i) {
		float inputValue = input(i);
		float exponential = std::exp(-inputValue);
		float newValue = exponential / ((1.0f + exponential) * (1.0f + exponential));
		derivatives(i) = newValue;
	}
	return derivatives;
}

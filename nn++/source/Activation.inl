#pragma once

template <size_t UNITS>
static Vec<float, UNITS> Activation::ReLU::func(const Vec<float, UNITS> &input) {
	Vec<float, UNITS> activation;
	for (size_t i = 0; i < UNITS; ++i) {
		float inputValue = input.get(i);
		float newValue = inputValue < 0.0f ? 0.0f : inputValue;
		activation.set(i, newValue);
	}
	return activation;
}
template <size_t UNITS>
static Vec<float, UNITS> Activation::ReLU::derivative(const Vec<float, UNITS> &input) {
	Vec<float, UNITS> derivatives;
	for (size_t i = 0; i < UNITS; ++i) {
		float inputValue = input.get(i);
		float newValue = inputValue < 0.0f ? 0.0f : 1.0f;
		derivatives.set(i, newValue);
	}
	return derivatives;
}

template <size_t UNITS>
static Vec<float, UNITS> Activation::LeakyReLU::func(const Vec<float, UNITS> &input) {
	Vec<float, UNITS> activation;
	for (size_t i = 0; i < UNITS; ++i) {
		float inputValue = input.get(i);
		float newValue = inputValue < 0.0f ? inputValue * 0.01f : inputValue;
		activation.set(i, newValue);
	}
	return activation;
}
template <size_t UNITS>
static Vec<float, UNITS> Activation::LeakyReLU::derivative(const Vec<float, UNITS> &input) {
	Vec<float, UNITS> derivatives;
	for (size_t i = 0; i < UNITS; ++i) {
		float inputValue = input.get(i);
		float newValue = inputValue < 0.0f ? 0.01f : 1.0f;
		derivatives.set(i, newValue);
	}
	return derivatives;
}

template <size_t UNITS>
static Vec<float, UNITS> Activation::Linear::func(const Vec<float, UNITS> &input) {
	return input;
}
template <size_t UNITS>
static Vec<float, UNITS> Activation::Linear::derivative(const Vec<float, UNITS> &input) {
	return Vec<float, UNITS>(1.0f);
}

template <size_t UNITS>
static Vec<float, UNITS> Activation::Sigmoid::func(const Vec<float, UNITS> &input) {
	Vec<float, UNITS> activation;
	for (size_t i = 0; i < UNITS; ++i) {
		float inputValue = input.get(i);
		float newValue = 1.0f / (1.0f + std::exp(-inputValue));
		activation.set(i, newValue);
	}
	return activation;
}
template <size_t UNITS>
static Vec<float, UNITS> Activation::Sigmoid::derivative(const Vec<float, UNITS> &input) {
	Vec<float, UNITS> derivatives;
	for (size_t i = 0; i < UNITS; ++i) {
		float inputValue = input.get(i);
		float exponential = std::exp(-inputValue);
		float newValue = exponential / ((1.0f + exponential) * (1.0f + exponential));
		derivatives.set(i, newValue);
	}
	return derivatives;
}

template <size_t UNITS>
static Vec<float, UNITS> Activation::Tanh::func(const Vec<float, UNITS> &input) {
	Vec<float, UNITS> activation;
	for (size_t i = 0; i < UNITS; ++i) {
		float inputValue = input.get(i);
		float newValue = 2.0f / (1.0f + std::exp(-2.0f * inputValue)) - 1.0f;
		activation.set(i, newValue);
	}
	return activation;
}
template <size_t UNITS>
static Vec<float, UNITS> Activation::Tanh::derivative(const Vec<float, UNITS> &input) {
	Vec<float, UNITS> derivatives;
	for (size_t i = 0; i < UNITS; ++i) {
		float inputValue = input.get(i);
		float exponential = std::exp(-2.0f * inputValue);
		float newValue = 4.0f * exponential /
			((1.0f + exponential) * (1.0f + exponential));
		derivatives.set(i, newValue);
	}
	return derivatives;
}

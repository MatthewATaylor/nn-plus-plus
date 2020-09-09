#pragma once

template <size_t INPUT_SIZE, size_t UNITS, typename ActivationType>
Dense<INPUT_SIZE, UNITS, ActivationType>::Dense() {
	std::random_device randomDevice;
	std::mt19937 generator(randomDevice());
	std::normal_distribution<> distribution(0, 1);
	for (size_t i = 1; i <= UNITS; ++i) {
		for (size_t j = 1; j <= INPUT_SIZE; ++j) {
			weights.set(i, j, (float) distribution(generator) * 0.01f);
		}
	}
}

template <size_t INPUT_SIZE, size_t UNITS, typename ActivationType>
Vec<float, UNITS> Dense<INPUT_SIZE, UNITS, ActivationType>::evaluate(
	const Vec<float, INPUT_SIZE> &input
) const {
	return ActivationType::func(weights * input + biases);
}
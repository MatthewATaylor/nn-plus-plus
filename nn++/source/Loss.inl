#pragma once

template <size_t OUTPUT_SIZE>
float Loss::MeanSquaredError::func(
	const Vec<float, OUTPUT_SIZE> &actual,
	const Vec<float, OUTPUT_SIZE> &prediction
) {
	float squaredErrorSum = 0.0f;
	for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
		squaredErrorSum += (
			(actual.get(i) - prediction.get(i)) *
			(actual.get(i) - prediction.get(i))
		);
	}
	return squaredErrorSum / OUTPUT_SIZE;
}
template <size_t OUTPUT_SIZE>
float Loss::MeanSquaredError::derivative(
	const Vec<float, OUTPUT_SIZE> &actual,
	const Vec<float, OUTPUT_SIZE> &prediction
) {
	float derivativeSum = 0.0f;
	for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
		derivativeSum += 2 * (actual.get(i) - prediction.get(i));
	}
	return derivativeSum / OUTPUT_SIZE;
}

template <size_t OUTPUT_SIZE>
float Loss::BinaryCrossEntropy::func(bool actual, float prediction) {
	if (actual) {
		return -std::log(prediction);
	}
	else {
		return -std::log(1.0f - prediction);
	}
}
template <size_t OUTPUT_SIZE>
float Loss::BinaryCrossEntropy::derivative(bool actual, float prediction) {
	if (actual) {
		return -1.0f / prediction;
	}
	else {
		return 1.0f / (1.0f - prediction);
	}
}

template <size_t OUTPUT_SIZE>
float Loss::CategoricalCrossEntropy::func(
	const Vec<bool, OUTPUT_SIZE> &actual,
	const Vec<float, OUTPUT_SIZE> &prediction
) {
	float lossTotal = 0.0f;
	for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
		if (actual.get(i)) {
			lossTotal -= std::log(prediction.get(i));
		}
		else {
			lossTotal -= std::log(1.0f - prediction.get(i));
		}
	}
	return lossTotal / OUTPUT_SIZE;
}
template <size_t OUTPUT_SIZE>
float Loss::CategoricalCrossEntropy::derivative(
	const Vec<bool, OUTPUT_SIZE> &actual,
	const Vec<float, OUTPUT_SIZE> &prediction
) {
	float derivativeTotal = 0.0f;
	for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
		if (actual.get(i)) {
			derivativeTotal -= 1.0f / prediction.get(i);
		}
		else {
			derivativeTotal += 1.0f / (1.0f - prediction.get(i));
		}
	}
	return derivativeTotal / OUTPUT_SIZE;
}

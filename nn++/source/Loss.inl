#pragma once

template <size_t OUTPUT_SIZE>
inline float Loss::MeanSquaredError::func(
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
inline Vec<float, OUTPUT_SIZE> Loss::MeanSquaredError::derivative(
	const Vec<float, OUTPUT_SIZE> &actual,
	const Vec<float, OUTPUT_SIZE> &prediction
) {
	Vec<float, OUTPUT_SIZE> derivatives;
	for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
		derivatives.set(i, (2 / OUTPUT_SIZE) * (actual.get(i) - prediction.get(i)));
	}
	return derivatives;
}

template <size_t OUTPUT_SIZE>
inline float Loss::BinaryCrossEntropy::func(
	const Vec<float, OUTPUT_SIZE> &actual,
	const Vec<float, OUTPUT_SIZE> &prediction
) {
	static_assert(
		OUTPUT_SIZE == 1,
		"Binary cross-entropy requires an output size of 1."
	);
	return
		actual.get(0) * -std::log(prediction.get(0)) +
		(1.0f - actual.get(0)) * -std::log(1.0f - prediction.get(0));
}
template <size_t OUTPUT_SIZE>
inline Vec<float, OUTPUT_SIZE> Loss::BinaryCrossEntropy::derivative(
	const Vec<float, OUTPUT_SIZE> &actual,
	const Vec<float, OUTPUT_SIZE> &prediction
) {
	static_assert(
		OUTPUT_SIZE == 1,
		"Binary cross-entropy requires an output size of 1."
	);
	Vec<float, OUTPUT_SIZE> derivatives(
		-actual.get(0) / prediction.get(0) +
		(1.0f - actual.get(0)) / (1.0f - prediction.get(0))
	);
	return derivatives;
}

template <size_t OUTPUT_SIZE>
inline float Loss::CategoricalCrossEntropy::func(
	const Vec<float, OUTPUT_SIZE> &actual,
	const Vec<float, OUTPUT_SIZE> &prediction
) {
	float lossTotal = 0.0f;
	for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
		lossTotal -=
			actual.get(i) * std::log(prediction.get(i)) +
			(1.0f - actual.get(i)) * std::log(1.0f - prediction.get(i));
	}
	return lossTotal / OUTPUT_SIZE;
}
template <size_t OUTPUT_SIZE>
inline Vec<float, OUTPUT_SIZE> Loss::CategoricalCrossEntropy::derivative(
	const Vec<float, OUTPUT_SIZE> &actual,
	const Vec<float, OUTPUT_SIZE> &prediction
) {
	Vec<float, OUTPUT_SIZE> derivatives;
	for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
		derivatives.set(
			i,
			-actual.get(i) / prediction.get(i) +
			(1.0f - actual.get(i)) / (1.0f - prediction.get(i))
		);
	}
	return derivatives;
}

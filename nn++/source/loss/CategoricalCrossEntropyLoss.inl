#pragma once

inline float CategoricalCrossEntropyLoss::func(
	const Vec<float> &actual, const Vec<float> &prediction
) const {
	float lossTotal = 0.0f;
	for (size_t i = 0; i < actual.size; ++i) {
		lossTotal -=
			actual(i) * std::log(prediction(i)) +
			(1.0f - actual(i)) * std::log(1.0f - prediction(i));
	}
	return lossTotal / actual.size;
}

inline Vec<float> CategoricalCrossEntropyLoss::derivative(
	const Vec<float> &actual, const Vec<float> &prediction
) const {
	Vec<float> derivatives(actual.size);
	for (size_t i = 0; i < actual.size; ++i) {
		derivatives(i) =
			-actual(i) / prediction(i) +
			(1.0f - actual(i)) / (1.0f - prediction(i));
	}
	return derivatives;
}

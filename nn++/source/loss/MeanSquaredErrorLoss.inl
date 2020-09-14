#pragma once

inline float MeanSquaredErrorLoss::func(
	const Vec<float> &actual, const Vec<float> &prediction
) const {
	float squaredErrorSum = 0.0f;
	for (size_t i = 0; i < actual.size; ++i) {
		squaredErrorSum += (
			(actual(i) - prediction(i)) *
			(actual(i) - prediction(i))
		);
	}
	return squaredErrorSum / actual.size;
}

inline Vec<float> MeanSquaredErrorLoss::derivative(
	const Vec<float> &actual, const Vec<float> &prediction
) const {
	Vec<float> derivatives(actual.size);
	for (size_t i = 0; i < actual.size; ++i) {
		derivatives(i) =  (2.0f / actual.size) * (prediction(i) - actual(i));
	}
	return derivatives;
}

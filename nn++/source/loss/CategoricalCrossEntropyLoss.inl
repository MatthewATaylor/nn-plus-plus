#pragma once

inline float CategoricalCrossEntropyLoss::func(
	const Vec<float> &actual, const Vec<float> &prediction
) const {
	float lossTotal = 0.0f;
	for (size_t i = 0; i < actual.size; ++i) {
		lossTotal -= actual(i) * std::log(prediction(i));
	}
	return lossTotal;
}

inline Vec<float> CategoricalCrossEntropyLoss::derivative(
	const Vec<float> &actual, const Vec<float> &prediction
) const {
	Vec<float> derivatives(actual.size);
	for (size_t i = 0; i < actual.size; ++i) {
		derivatives(i) = -actual(i) / (prediction(i));
	}
	return derivatives;
}

#pragma once

inline float CategoricalCrossEntropyLoss::func(
	const Mat<float> &actual, const Mat<float> &prediction
) const {
	float lossTotal = 0.0f;
	for (size_t i = 0; i < actual.rows; ++i) {
		for (size_t j = 0; j < actual.cols; ++j) {
			lossTotal -= actual(i, j) * std::log(prediction(i, j) + 0.0000000001f);
		}
	}
	return lossTotal / actual.cols;
}

inline Mat<float> CategoricalCrossEntropyLoss::derivative(
	const Mat<float> &actual, const Mat<float> &prediction
) const {
	Mat<float> derivatives(actual.rows, actual.cols);
	for (size_t i = 0; i < actual.rows; ++i) {
		for (size_t j = 0; j < actual.cols; ++j) {
			derivatives(i, j) = -actual(i, j) / (prediction(i, j));
		}
	}
	return derivatives;
}

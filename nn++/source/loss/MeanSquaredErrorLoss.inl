#pragma once

inline float MeanSquaredErrorLoss::func(
	const Mat<float> &actual, const Mat<float> &prediction
) const {
	float squaredErrorSum = 0.0f;
	for (size_t i = 0; i < actual.rows; ++i) {
		for (size_t j = 0; j < actual.cols; ++j) {
			squaredErrorSum +=
				(actual(i, j) - prediction(i, j)) *
				(actual(i, j) - prediction(i, j));
		}
	}
	return squaredErrorSum / actual.rows / actual.cols;
}

inline Mat<float> MeanSquaredErrorLoss::derivative(
	const Mat<float> &actual, const Mat<float> &prediction
) const {
	Mat<float> derivatives(actual.rows, actual.cols);
	for (size_t i = 0; i < actual.rows; ++i) {
		for (size_t j = 0; j < actual.cols; ++j) {
			derivatives(i, j) =
				(2.0f / actual.rows / actual.cols) *
				(prediction(i, j) - actual(i, j));
		}
	}
	return derivatives;
}

#pragma once

inline ReLUActivation::ReLUActivation() {
	weightFactor = 2.0f;
}

inline Mat<float> ReLUActivation::func(const Mat<float> &input) const {
	Mat<float> activation(input.rows, input.cols);
	for (size_t i = 0; i < input.rows; ++i) {
		for (size_t j = 0; j < input.cols; ++j) {
			float inputValue = input(i, j);
			activation(i, j) = inputValue < 0.0f ? 0.0f : inputValue;
		}
	}
	return activation;
}

inline Mat<float> ReLUActivation::derivative(const Mat<float> &input) const {
	Mat<float> derivatives(input.rows, input.cols);
	for (size_t i = 0; i < input.rows; ++i) {
		for (size_t j = 0; j < input.cols; ++j) {
			float inputValue = input(i, j);
			derivatives(i, j) = inputValue < 0.0f ? 0.0f : 1.0f;
		}
	}
	return derivatives;
}

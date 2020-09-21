#pragma once

inline Mat<float> TanhActivation::func(const Mat<float> &input) const {
	Mat<float> activation(input.rows, input.cols);
	for (size_t i = 0; i < input.rows; ++i) {
		for (size_t j = 0; j < input.cols; ++j) {
			float inputValue = input(i, j);
			activation(i, j) = 2.0f / (1.0f + std::exp(-2.0f * inputValue)) - 1.0f;
		}
	}
	return activation;
}

inline Mat<float> TanhActivation::derivative(const Mat<float> &input) const {
	Mat<float> derivatives(input.rows, input.cols);
	for (size_t i = 0; i < input.rows; ++i) {
		for (size_t j = 0; j < input.cols; ++j) {
			float inputValue = input(i, j);
			float exponential = std::exp(-2.0f * inputValue);
			derivatives(i, j) = 4.0f * exponential / ((1.0f + exponential) * (1.0f + exponential));
		}
	}
	return derivatives;
}

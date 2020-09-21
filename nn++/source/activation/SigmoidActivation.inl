#pragma once

inline Mat<float> SigmoidActivation::func(const Mat<float> &input) const {
	Mat<float> activation(input.rows, input.cols);
	for (size_t i = 0; i < input.rows; ++i) {
		for (size_t j = 0; j < input.cols; ++j) {
			float inputValue = input(i, j);
			activation(i, j) = 1.0f / (1.0f + std::exp(-inputValue));
		}
	}
	return activation;
}

inline Mat<float> SigmoidActivation::derivative(const Mat<float> &input) const {
	Mat<float> derivatives(input.rows, input.cols);
	for (size_t i = 0; i < input.rows; ++i) {
		for (size_t j = 0; j < input.cols; ++j) {
			float inputValue = input(i, j);
			float exponential = std::exp(-inputValue);
			derivatives(i, j) = exponential / ((1.0f + exponential) * (1.0f + exponential));
		}
	}
	return derivatives;
}

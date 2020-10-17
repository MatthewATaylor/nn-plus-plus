#pragma once

inline LeakyReLUActivation::LeakyReLUActivation(float leakSlope) :
	leakSlope(leakSlope) {}

inline Mat<float> LeakyReLUActivation::func(const Mat<float> &input) const {
	Mat<float> activation(input.rows, input.cols);
	for (size_t i = 0; i < input.rows; ++i) {
		for (size_t j = 0; j < input.cols; ++j) {
			float inputValue = input(i, j);
			activation(i, j) = inputValue < 0.0f ? inputValue * leakSlope : inputValue;
		}
	}
	return activation;
}

inline Mat<float> LeakyReLUActivation::derivative(const Mat<float> &input) const {
	Mat<float> derivatives(input.rows, input.cols);
	for (size_t i = 0; i < input.rows; ++i) {
		for (size_t j = 0; j < input.cols; ++j) {
			float inputValue = input(i, j);
			derivatives(i, j) = inputValue < 0.0f ? leakSlope : 1;
		}
	}
	return derivatives;
}

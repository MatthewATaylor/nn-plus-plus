#pragma once

inline Mat<float> LinearActivation::func(const Mat<float> &input) const {
	return input;
}

inline Mat<float> LinearActivation::derivative(const Mat<float> &input) const {
	return Mat<float>(input.rows, input.cols, 1.0f);
}

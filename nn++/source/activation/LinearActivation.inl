#pragma once

inline Vec<float> LinearActivation::func(const Vec<float> &input) const {
	return input;
}

inline Vec<float> LinearActivation::derivative(const Vec<float> &input) const {
	return Vec<float>(input.size, 1.0f);
}

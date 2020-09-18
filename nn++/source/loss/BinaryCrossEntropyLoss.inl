#pragma once

inline float BinaryCrossEntropyLoss::func(
	const Vec<float> &actual, const Vec<float> &prediction
) const {
	return
		-actual(0) * std::log(prediction(0) + 0.000000000000000000001f) -
		(1.0f - actual(0)) * std::log(1.0f - prediction(0) + 0.000000000000000000001f);
}

inline Vec<float> BinaryCrossEntropyLoss::derivative(
	const Vec<float> &actual, const Vec<float> &prediction
) const {
	return Vec<float>(
		1,
		-actual(0) / prediction(0) +
			(1.0f - actual(0)) / (1.0f - prediction(0))
	);
}

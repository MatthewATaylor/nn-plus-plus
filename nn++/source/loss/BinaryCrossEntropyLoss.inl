#pragma once

inline float BinaryCrossEntropyLoss::func(
	const Mat<float> &actual, const Mat<float> &prediction
) const {
	float totalLoss = 0.0f;
	for (size_t i = 0; i < actual.cols; ++i) { //For each input in batch
		totalLoss +=
			-actual(0, i) * std::log(prediction(0, i) + 0.0000000001f) -
			(1.0f - actual(0, i)) * std::log(1.0f - prediction(0, i) + 0.0000000001f);
	}
	return totalLoss / actual.cols;
}

inline Mat<float> BinaryCrossEntropyLoss::derivative(
	const Mat<float> &actual, const Mat<float> &prediction
) const {
	Mat<float> derivatives(1, actual.cols);
	for (size_t i = 0; i < actual.cols; ++i) {
		derivatives(0, i) =
			-actual(0, i) / prediction(0, i) +
			(1.0f - actual(0, i)) / (1.0f - prediction(0, i));
	}
	return derivatives;
}

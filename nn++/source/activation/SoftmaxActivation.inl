#pragma once

inline Mat<float> SoftmaxActivation::func(const Mat<float> &input) const {
	Mat<float> activation(input.rows, input.cols);
	for (size_t i = 0; i < input.cols; ++i) { //For each input in batch
		float exponentialSum = 0.0f;
		for (size_t j = 0; j < input.rows; ++j) {
			exponentialSum += std::exp(input(j, i));
		}
		for (size_t j = 0; j < input.rows; ++j) {
			activation(j, i) = std::exp(input(j, i)) / (exponentialSum + 0.000000001f);
		}
	}
	return activation;
}

inline Mat<float> SoftmaxActivation::derivative(const Mat<float> &input) const {
	throw std::exception(
		"Softmax may only be used in an output layer and with categorical cross-entropy loss"
	);
}

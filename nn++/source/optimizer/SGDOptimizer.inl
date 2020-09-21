#pragma once

inline SGDOptimizer::SGDOptimizer(float learningRate) :
	LEARNING_RATE(learningRate) {}

inline void SGDOptimizer::updateLayer(Dense *layer, const Mat<float> &prevNodes, unsigned int timestep) {
	//Take sum of all dC/dW and dC/dB in batch
	Mat<float> dwTotal(layer->units, layer->inputSize, 0.0f);
	Vec<float> dbTotal(layer->units, 0.0f);
	for (size_t i = 0; i < layer->errors.cols; ++i) { //For each column in batch
		Mat<float> errorCol(layer->units, 1);
		for (size_t j = 0; j < layer->units; ++j) {
			dbTotal(j) += layer->errors(j, i);
			errorCol(j, 0) = layer->errors(j, i);
		}

		Mat<float> prevActivationsRow(1, prevNodes.rows);
		for (size_t j = 0; j < prevNodes.rows; ++j) {
			prevActivationsRow(0, j) = prevNodes(j, i);
		}

		dwTotal += errorCol * prevActivationsRow;
	}

	//Update weights and biases based on average dC/dW and dC/dB in batch
	layer->weights -= dwTotal / float(layer->errors.cols) * LEARNING_RATE;
	layer->biases -= dbTotal / float(layer->errors.cols) * LEARNING_RATE;
}

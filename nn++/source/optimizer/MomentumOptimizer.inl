#pragma once

inline MomentumOptimizer::MomentumOptimizer(float learningRate, float beta) :
	LEARNING_RATE(learningRate), BETA(beta) {}

inline void MomentumOptimizer::updateLayer(Dense *layer, const Mat<float> &prevNodes, unsigned int timestep) {
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
	
	//Update weight and bias V values based on average dC/dW and dC/dB in batch
	layer->weightV =
		(layer->weightV * BETA) +
		(dwTotal / float(layer->errors.cols) * (1.0f - BETA));
	layer->biasV =
		(layer->biasV * BETA) +
		(dbTotal / float(layer->errors.cols) * (1.0f - BETA));

	layer->weights -= layer->weightV * LEARNING_RATE;
	layer->biases -= layer->biasV * LEARNING_RATE;
}

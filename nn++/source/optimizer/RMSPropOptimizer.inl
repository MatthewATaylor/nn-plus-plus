#pragma once

inline RMSPropOptimizer::RMSPropOptimizer(float learningRate, float beta) :
	LEARNING_RATE(learningRate), BETA(beta) {}

inline void RMSPropOptimizer::updateLayer(Dense *layer, const Mat<float> &prevNodes, unsigned int timestep) {
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

	Mat<float> dwAvg = dwTotal / (float) layer->errors.cols;
	Vec<float> dbAvg = dbTotal / (float) layer->errors.cols;

	//Update weight and bias S values based on average dC/dW and dC/dB in batch
	layer->weightS =
		(layer->weightS * BETA) +
		(dwAvg.powByElements(2) * (1.0f - BETA));
	layer->biasS =
		(layer->biasS * BETA) +
		(dbAvg * dbAvg * (1.0f - BETA));

	layer->weights -=
		(dwAvg * LEARNING_RATE).divideByElements(
			layer->weightS.powByElements(0.5f) + 0.000000001f
		);
	layer->biases -=
		(dbAvg * LEARNING_RATE) /
		(layer->biasS.pow(0.5f) + 0.000000001f);
}

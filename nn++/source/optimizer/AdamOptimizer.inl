#pragma once

inline AdamOptimizer::AdamOptimizer(float learningRate, float beta1, float beta2) :
	LEARNING_RATE(learningRate), BETA_1(beta1), BETA_2(beta2) {}

inline void AdamOptimizer::updateLayer(Dense *layer, const Mat<float> &prevNodes, unsigned int timestep) {
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

	//Update V values (momentum)
	layer->weightV =
		(layer->weightV * BETA_1) +
		(dwAvg * (1.0f - BETA_1));
	layer->biasV =
		(layer->biasV * BETA_1) +
		(dbAvg * (1.0f - BETA_1));
	
	//Update S values (RMSProp)
	layer->weightS =
		(layer->weightS * BETA_2) +
		(dwAvg.powByElements(2) * (1.0f - BETA_2));
	layer->biasS =
		(layer->biasS * BETA_2) +
		(dbAvg * dbAvg * (1.0f - BETA_2));
	
	//Perform initial bias correction
	const float BETA_1_POW = (float)std::pow(BETA_1, timestep);
	Mat<float> correctedWeightV = layer->weightV / (1.0f - BETA_1_POW);
	Vec<float> correctedBiasV = layer->biasV / (1.0f - BETA_1_POW);

	const float BETA_2_POW = (float)std::pow(BETA_2, timestep);
	Mat<float> correctedWeightS = layer->weightS / (1.0f - BETA_2_POW);
	Vec<float> correctedBiasS = layer->biasS / (1.0f - BETA_2_POW);

	//Adjust weights and biases
	layer->weights -=
		(correctedWeightV * LEARNING_RATE).divideByElements(
			correctedWeightS.powByElements(0.5f) + 0.000000001f
		);
	layer->biases -=
		(correctedBiasV * LEARNING_RATE) /
		(correctedBiasS.pow(0.5f) + 0.000000001f);
}

#pragma once

inline RMSPropOptimizer::RMSPropOptimizer(float learningRate, float beta) :
	LEARNING_RATE(learningRate), BETA(beta) {}

inline void RMSPropOptimizer::updateLayer(Dense *layer, const Vec<float> &prevNodes, unsigned int timestep) {
	Mat<float> errors(layer->errors);
	Mat<float> prevActivations(prevNodes, false);
	Mat<float> weightError = errors * prevActivations;

	layer->weightS =
		(layer->weightS * BETA) +
		(weightError.powByElements(2) * (1.0f - BETA));
	layer->biasS =
		(layer->biasS * BETA) +
		(layer->errors * layer->errors * (1.0f - BETA));

	layer->weights -=
		(weightError * LEARNING_RATE).divideByElements(
			layer->weightS.powByElements(0.5f) + 0.000000001f
		);
	layer->biases -=
		(layer->errors * LEARNING_RATE) /
		(layer->biasS.pow(0.5f) + 0.000000001f);
}

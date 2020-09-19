#pragma once

inline MomentumOptimizer::MomentumOptimizer(float learningRate, float beta) :
	LEARNING_RATE(learningRate), BETA(beta) {}

inline void MomentumOptimizer::updateLayer(Dense *layer, const Vec<float> &prevNodes, unsigned int timestep) {
	Mat<float> errors(layer->errors);
	Mat<float> prevActivations(prevNodes, false);
	Mat<float> weightError = errors * prevActivations;

	layer->weightV = (layer->weightV * BETA) + (weightError * (1.0f - BETA));
	layer->biasV = (layer->biasV * BETA) + (layer->errors * (1.0f - BETA));

	layer->weights -= layer->weightV * LEARNING_RATE;
	layer->biases -= layer->biasV * LEARNING_RATE;
}

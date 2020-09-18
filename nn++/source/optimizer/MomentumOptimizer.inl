#pragma once

inline MomentumOptimizer::MomentumOptimizer(float learningRate, float beta) :
	LEARNING_RATE(learningRate), BETA(beta) {}

inline void MomentumOptimizer::updateLayer(Dense *layer, const Vec<float> &prevNodes) {
	Mat<float> errors(layer->errors);
	Mat<float> prevActivations(prevNodes, false);
	Mat<float> weightError = errors * prevActivations;

	layer->weightVelocity = (layer->weightVelocity * BETA) + (weightError * (1.0f - BETA));
	layer->biasVelocity = (layer->biasVelocity * BETA) + (layer->errors * (1.0f - BETA));

	layer->weights -= layer->weightVelocity * LEARNING_RATE;
	layer->biases -= layer->biasVelocity * LEARNING_RATE;
}

#pragma once

inline SGDOptimizer::SGDOptimizer(float learningRate) :
	LEARNING_RATE(learningRate) {}

inline void SGDOptimizer::updateLayer(Dense *layer, const Vec<float> &prevNodes, unsigned int timestep) {
	Mat<float> outputErrors(layer->errors);
	Mat<float> prevActivations(prevNodes, false);
	Mat<float> weightError = outputErrors * prevActivations;

	layer->weights -= weightError * LEARNING_RATE;
	layer->biases -= layer->errors * LEARNING_RATE;
}

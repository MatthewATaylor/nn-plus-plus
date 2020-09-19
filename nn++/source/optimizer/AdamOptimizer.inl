#pragma once

inline AdamOptimizer::AdamOptimizer(float learningRate, float beta1, float beta2) :
	LEARNING_RATE(learningRate), BETA_1(beta1), BETA_2(beta2) {}

inline void AdamOptimizer::updateLayer(Dense *layer, const Vec<float> &prevNodes, unsigned int timestep) {
	Mat<float> errors(layer->errors);
	Mat<float> prevActivations(prevNodes, false);
	Mat<float> weightError = errors * prevActivations;

	//Update V values (momentum)
	layer->weightV = layer->weightV * BETA_1 + weightError * (1.0f - BETA_1);
	layer->biasV = layer->biasV * BETA_1 + layer->errors * (1.0f - BETA_1);

	//Update S values (RMSProp)
	layer->weightS =
		(layer->weightS * BETA_2) +
		(weightError.powByElements(2) * (1.0f - BETA_2));
	layer->biasS =
		(layer->biasS * BETA_2) +
		(layer->errors * layer->errors * (1.0f - BETA_2));

	//Perform initial bias correction
	const float BETA_1_POW = (float) std::pow(BETA_1, timestep);
	Mat<float> correctedWeightV = layer->weightV / (1.0f - BETA_1_POW);
	Vec<float> correctedBiasV = layer->biasV / (1.0f - BETA_1_POW);
	
	const float BETA_2_POW = (float) std::pow(BETA_2, timestep);
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

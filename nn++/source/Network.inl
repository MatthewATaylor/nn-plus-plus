#pragma once

inline void Network::updateWeights(
	size_t layerIndex, const Vec<float> &prevNodes, float learningRate
) {
	Mat<float> outputErrors(layers[layerIndex]->errors);
	Mat<float> prevActivations = Mat<float>(prevNodes, false);
	Mat<float> weightError = outputErrors * prevActivations;
	layers[layerIndex]->weights -= weightError * learningRate;
}

inline Network::Network() {}
inline Network::Network(std::initializer_list<Dense*> layers) :
	layers(layers) {}

inline void Network::append(Dense *layer) {
	layers.push_back(layer);
}

inline Vec<float> Network::evaluate(const Vec<float> &input) {
	Vec<float> result = input;
	for (size_t i = 0; i < layers.size(); ++i) {
		layers[i]->evaluate(result);
		result = layers[i]->activations;
	}
	return result;
}

inline float Network::getLoss(
	const Vec<float> &input, const Vec<float> &target, const Loss *loss
) {
	Vec<float> prediction = evaluate(input);
	return loss->func(target, prediction);
}

inline void Network::train(
	const Vec<float> *inputs, const Vec<float> *targets, size_t dataSize,
	const Loss *loss, float learningRate, unsigned int epochs
) {
	if (layers.size() == 0) {
		return;
	}

	for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
		float totalLoss = 0.0f;
		for (size_t i = 0; i < dataSize; ++i) {
			//Evaluate all layers' activations and weighted inputs
			Vec<float> prediction = evaluate(inputs[i]);
			std::cout << inputs[i] << "    " << prediction << "\n";
			totalLoss += loss->func(targets[i], prediction);

			//Calculate output layer error
			size_t outputLayerIndex = layers.size() - 1;
			Dense *outputLayer = layers[outputLayerIndex];
			outputLayer->errors = loss->derivative(targets[i], prediction);
			outputLayer->errors *= outputLayer->activationFuncDerivative();

			//Backpropagate error
			for (size_t j = layers.size() - 2; j != static_cast<size_t>(-1); --j) {
				Dense *layer = layers[j];
				Dense *prevLayer = layers[j + 1];
				layer->errors = prevLayer->weights.transpose() * prevLayer->errors;
				layer->errors *= layer->activationFuncDerivative();
			}

			//Update initial layer weights and biases
			updateWeights(0, inputs[i], learningRate);
			layers[0]->biases -= layers[0]->errors * learningRate;

			//Update other weights and biases
			for (size_t i = 1; i < layers.size(); ++i) {
				updateWeights(i, layers[i - 1]->activations, learningRate);
				layers[i]->biases -= layers[i]->errors * learningRate;
			}
		}
		std::cout << "Epoch " << epoch << "\n";
		std::cout << "    Loss: " << totalLoss / dataSize << "\n";
	}
}

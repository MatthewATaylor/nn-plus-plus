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

	size_t outputLayerIndex = layers.size() - 1;
	Dense *outputLayer = layers[outputLayerIndex];
	bool isSoftmaxOutput = false;

	if (typeid(*outputLayer->getActivation()) == typeid(SoftmaxActivation)) {
		isSoftmaxOutput = true;
		if (typeid(*loss) != typeid(CategoricalCrossEntropyLoss)) {
			throw std::exception(
				"Categorical cross-entropy loss must be used with an output layer softmax activation."
			);
		}
	}

	for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
		float totalLoss = 0.0f;
		for (size_t i = 0; i < dataSize; ++i) {
			//Evaluate all layers' activations and weighted inputs
			Vec<float> prediction = evaluate(inputs[i]);
			totalLoss += loss->func(targets[i], prediction);

			//Calculate output layer error
			if (isSoftmaxOutput) {
				//Softmax special case: combine categorical and softmax derivatives
				outputLayer->errors = prediction - targets[i];
			}
			else {
				outputLayer->errors = loss->derivative(targets[i], prediction);
				outputLayer->errors *= outputLayer->activationFuncDerivative();
			}

			//Backpropagate error
			for (size_t j = layers.size() - 2; j != static_cast<size_t>(-1); --j) {
				Dense *layer = layers[j];
				Dense *prevLayer = layers[j + 1];
				layer->errors = prevLayer->weights.transpose() * prevLayer->errors;
				layer->errors *= layer->activationFuncDerivative();
				std::cout << layer->errors << "\n";
			}
			//std::cin.get();

			//Update initial layer weights and biases
			updateWeights(0, inputs[i], learningRate);
			layers[0]->biases -= layers[0]->errors * learningRate;

			//Update other weights and biases
			for (size_t j = 1; j < layers.size(); ++j) {
				updateWeights(j, layers[j - 1]->activations, learningRate);
				layers[j]->biases -= layers[j]->errors * learningRate;
			}
		}
		std::cout << "Epoch " << epoch << "\n";
		std::cout << "    Loss: " << totalLoss / dataSize << "\n";
		//display();
		std::cin.get();
	}
}

void Network::display() const {
	for (size_t i = 0; i < layers.size(); ++i) {
		std::cout << "Layer " << i << "\n";
		std::cout << "Weights:\n";
		std::cout << layers[i]->weights << "\n";
		std::cout << "Biases:\n";
		std::cout << layers[i]->biases << "\n\n";
	}
	std::cout << "\n";
}

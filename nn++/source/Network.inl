#pragma once

inline Network::Network() {}
inline Network::Network(std::initializer_list<Dense*> layers) :
	layers(layers) {}

inline void Network::append(Dense *layer) {
	layers.push_back(layer);
}

inline Mat<float> Network::evaluate(const Mat<float> &inputBatch) {
	Mat<float> result = inputBatch;
	for (size_t i = 0; i < layers.size(); ++i) {
		layers[i]->evaluate(result);
		result = layers[i]->activations;
	}
	return result;
}

inline float Network::getLoss(
	const Mat<float> &input, const Mat<float> &target, const Loss *loss
) {
	Mat<float> prediction = evaluate(input);
	return loss->func(target, prediction);
}

inline void Network::setAccuracyFunc(
	std::function<bool(const Mat<float> &prediction, const Mat<float> &target)> accuracyFunc
) {
	accuracyEnabled = true;
	this->accuracyFunc = accuracyFunc;
}

inline void Network::train(
	const Vec<float> *inputs, const Vec<float> *targets, size_t dataSize,
	const Loss *loss, Optimizer *optimizer, unsigned int epochs,
	size_t batchSize, bool shuffle,
	const Vec<float> *valInputs, const Vec<float> *valTargets, size_t valSize
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

	size_t *indices = new size_t[dataSize];
	for (size_t i = 0; i < dataSize; ++i) {
		indices[i] = i;
	}

	size_t dataUtilized = batchSize * (dataSize / batchSize);

	for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
		std::cout << "Epoch " << epoch << "\n";
		
		if (shuffle) {
			std::random_device randomDevice;
			std::mt19937 generator(randomDevice());
			std::shuffle(indices, indices + dataSize, generator);
		}
		
		float totalLoss = 0.0f;
		size_t correctPredictions = 0;
		for (size_t indexNum = 0; indexNum <= dataSize - batchSize; indexNum += batchSize) {
			//Generate batched data
			size_t inputSize = inputs[0].size;
			size_t targetSize = targets[0].size;
			Mat<float> inputBatch(inputSize, batchSize);
			Mat<float> targetBatch(targetSize, batchSize);
			for (size_t j = 0; j < batchSize; ++j) {
				size_t i = indices[indexNum + j]; //Use potentially shuffled indices
				for (size_t k = 0; k < inputSize; ++k) {
					inputBatch(k, j) = inputs[i](k);
				}
				for (size_t k = 0; k < targetSize; ++k) {
					targetBatch(k, j) = targets[i](k);
				}
			}

			//Evaluate all layers' activations and weighted inputs
			Mat<float> prediction = evaluate(inputBatch);
			totalLoss += loss->func(targetBatch, prediction);

			//Determine accuracy
			if (accuracyEnabled) {
				for (size_t i = 0; i < batchSize; ++i) {
					Mat<float> predictionCol = Mat<float>(prediction.getCol(i));
					Mat<float> targetCol = Mat<float>(targetBatch.getCol(i));
					if (accuracyFunc(predictionCol, targetCol)) {
						++correctPredictions;
					}
				}
			}

			//Calculate output layer error
			if (isSoftmaxOutput) {
				//Softmax special case: combine categorical and softmax derivatives
				outputLayer->errors = prediction - targetBatch;
			}
			else {
				outputLayer->errors = loss->derivative(targetBatch, prediction);
				outputLayer->errors = outputLayer->errors.multiplyByElements(
					outputLayer->activationFuncDerivative()
				);
			}

			//Backpropagate error
			for (size_t j = layers.size() - 2; j != static_cast<size_t>(-1); --j) {
				Dense *layer = layers[j];
				Dense *prevLayer = layers[j + 1];
				layer->errors = prevLayer->weights.transpose() * prevLayer->errors;
				layer->errors = layer->errors.multiplyByElements(
					layer->activationFuncDerivative()
				);
			}

			//Update initial layer weights and biases
			optimizer->updateLayer(layers[0], inputBatch, epoch + 1);

			//Update other weights and biases
			for (size_t j = 1; j < layers.size(); ++j) {
				optimizer->updateLayer(layers[j], layers[j - 1]->activations, epoch + 1);
			}
		}

		std::cout << "    Loss: " << totalLoss / dataSize << "\n";
		if (accuracyEnabled) {
			std::cout << "    Accuracy: " << float(correctPredictions) / dataUtilized << "\n";
		}
		
		if (valInputs && valTargets) {
			float totalValLoss = 0.0f;
			size_t correctValPredictions = 0;
			for (size_t i = 0; i < valSize; ++i) {
				//Loss
				Mat<float> prediction = evaluate(Mat<float>(valInputs[i]));
				totalValLoss += loss->func(valTargets[i], prediction);

				//Accuracy
				if (accuracyEnabled) {
					if (accuracyFunc(prediction, Mat<float>(valTargets[i]))) {
						++correctValPredictions;
					}
				}
			}
			std::cout << "    Validation Loss: " << totalValLoss / valSize << "\n";

			if (accuracyEnabled) {
				std::cout << "    Validation Accuracy: " << float(correctValPredictions) / valSize << "\n";
			}
		}
	}

	delete[] indices;
}

const Dense *Network::getLayer(size_t index) const {
	return layers[index];
}

void Network::display() const {
	for (size_t i = 0; i < layers.size(); ++i) {
		std::cout << "Layer " << i << "\n";
		std::cout << "Weights:\n";
		std::cout << layers[i]->weights << "\n";
		std::cout << "Biases:\n";
		std::cout << layers[i]->biases << "\n";
		std::cout << "Errors:\n";
		std::cout << layers[i]->errors << "\n\n";
	}
	std::cout << "\n";
}

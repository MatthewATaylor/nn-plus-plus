#pragma once

#include <iostream>
#include <fstream>
#include <string>

#include "activation/ReLUActivation.h"
#include "activation/SoftmaxActivation.h"
#include "loss/CategoricalCrossEntropyLoss.h"
#include "optimizer/MomentumOptimizer.h"
#include "Dense.h"
#include "Network.h"
#include "math/Vec.h"

const size_t IMG_SIZE = 28;
const size_t IMG_PIXELS = IMG_SIZE * IMG_SIZE;
const size_t NUM_LABELS = 10;

void loadImages(Vec<float> *images, size_t numImages, const std::string &path) {
	std::ifstream fin(path, std::ios::in | std::ios::binary);
	fin.seekg(16);

	for (size_t i = 0; i < numImages; ++i) {
		for (size_t j = 0; j < IMG_PIXELS; ++j) {
			unsigned char pixel;
			fin.read(reinterpret_cast<char*>(&pixel), 1);
			images[i](j) = static_cast<float>(pixel) / 255.0f;
		}
	}
}

void loadLabels(Vec<float> *labels, size_t numLabels, const std::string &path) {
	std::ifstream fin(path, std::ios::in | std::ios::binary);
	fin.seekg(8);

	for (size_t i = 0; i < numLabels; ++i) {
		unsigned char label;
		fin.read(reinterpret_cast<char*>(&label), 1);
		labels[i](label) = 1.0f;
	}
}

int main() {
	const size_t TRAIN_SIZE = 1000;
	const size_t TEST_SIZE = 10;
	const std::string DATA_DIR = "../nn++/source/examples/mnist/data";

	std::cout << "Loading data...\n";

	//Load images
	Vec<float> *trainImages = new Vec<float>[TRAIN_SIZE];
	for (size_t i = 0; i < TRAIN_SIZE; ++i) {
		trainImages[i] = Vec<float>(IMG_PIXELS);
	}
	Vec<float> *testImages = new Vec<float>[TEST_SIZE];
	for (size_t i = 0; i < TEST_SIZE; ++i) {
		testImages[i] = Vec<float>(IMG_PIXELS);
	}
	loadImages(trainImages, TRAIN_SIZE, DATA_DIR + "/train-images.idx3-ubyte");
	loadImages(testImages, TEST_SIZE, DATA_DIR + "/test-images.idx3-ubyte");

	//Load labels
	Vec<float> *trainLabels = new Vec<float>[TRAIN_SIZE];
	for (size_t i = 0; i < TRAIN_SIZE; ++i) {
		trainLabels[i] = Vec<float>(NUM_LABELS, 0.0f);
	}
	Vec<float> *testLabels = new Vec<float>[TEST_SIZE];
	for (size_t i = 0; i < TEST_SIZE; ++i) {
		testLabels[i] = Vec<float>(NUM_LABELS, 0.0f);
	}
	loadLabels(trainLabels, TRAIN_SIZE, DATA_DIR + "/train-labels.idx1-ubyte");
	loadLabels(testLabels, TEST_SIZE, DATA_DIR + "/test-labels.idx1-ubyte");

	//Generate model
	ReLUActivation relu;
	SoftmaxActivation softmax;

	Dense dense1(IMG_PIXELS, 32, &relu);
	Dense dense2(32, 64, &relu);
	Dense dense3(64, NUM_LABELS, &softmax);

	Network network{
		&dense1,
		&dense2,
		&dense3
	};

	CategoricalCrossEntropyLoss loss;
	MomentumOptimizer optimizer(0.001f, 0.9f);

	std::cout << "\nTraining model...\n";
	network.train(trainImages, trainLabels, TRAIN_SIZE, &loss, &optimizer, 25);

	std::cout << "\nMaking predictions...\n";
	for (size_t i = 0; i < TEST_SIZE; ++i) {
		std::cout << "Prediction: " << network.evaluate(testImages[i]) << "\n";
		std::cout << "Target: " << testLabels[i] << "\n";
		std::cin.get();
	}

	std::cin.get();
	return 0;
}

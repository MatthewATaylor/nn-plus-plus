#pragma once

#include <iostream>
#include <fstream>
#include <string>

#include "activation/LeakyReLUActivation.h"
#include "activation/SoftmaxActivation.h"
#include "loss/CategoricalCrossEntropyLoss.h"
#include "optimizer/AdamOptimizer.h"
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

bool accuracyFunc(const Mat<float> &prediction, const Mat<float> &target) {
	size_t predictedCategory = prediction.maxRowByCol()(0);
	size_t actualCategory = target.maxRowByCol()(0);
	return predictedCategory == actualCategory;
}

int main() {
	const size_t TRAIN_SIZE = 1000;
	const size_t TEST_SIZE = 100;
	const std::string DATA_DIR = "../nn++/source/examples/mnist/data";

	std::cout << "Loading data...\n\n";

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

	//Display train data
	for (size_t i = 0; i < 5; ++i) {
		std::string imageStr = "";
		std::string simplifiedImageStr = "";

		//Image
		for (size_t row = 0; row < IMG_SIZE; ++row) {
			for (size_t col = 0; col < IMG_SIZE; ++col) {
				size_t index = col + IMG_SIZE * row;
				
				int pixelValue = int(trainImages[i](index) * 255);
				std::string pixelStr = std::to_string(pixelValue);
				
				while (pixelStr.size() < 3) {
					pixelStr.insert(pixelStr.begin(), '0');
				}
				imageStr += pixelStr + ' ';

				if (pixelValue) {
					simplifiedImageStr += '#';
				}
				else {
					simplifiedImageStr += ' ';
				}
			}
			imageStr += "\n";
			simplifiedImageStr += "\n";
		}
		std::cout << imageStr << simplifiedImageStr;

		//Label
		size_t number = Mat<float>(trainLabels[i]).maxRowByCol()(0);
		std::cout << "Label: " << number << "\n";

		std::cin.get();
	}

	//Activations
	LeakyReLUActivation leakyRelu;
	SoftmaxActivation softmax;
		//LinearActivation linear;
		//ReLUActivation relu;
		//SigmoidActivation sigmoid;
		//TanhActivation tanh;

	//Generate model
	Network network{
		new Dense(IMG_PIXELS, 32, &leakyRelu),	//32 units
		new Dense(32, 64, &leakyRelu),			//64 units
		new Dense(64, NUM_LABELS, &softmax)		//10 units
	};

	std::cout << "\nTraining model...\n";
	network.setAccuracyFunc(accuracyFunc);
	CategoricalCrossEntropyLoss loss;
		//BinaryCrossEntropyLoss loss;
		//MeanSquaredErrorLoss loss;
	AdamOptimizer optimizer; 
		//MomentumOptimizer optimizer;
		//RMSPropOptimizer optimizer;
		//SGDOptimizer optimizer(0.001f);

	network.train(
		trainImages, trainLabels, TRAIN_SIZE,
		&loss, &optimizer, 50, 64, true,
		testImages, testLabels, TEST_SIZE
	);

	std::cout << "\nPress enter to exit...";
	std::cin.get();
	return 0;
}

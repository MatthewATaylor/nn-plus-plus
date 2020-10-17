#pragma once

#include <vector>
#include <functional>
#include <initializer_list>
#include <typeinfo>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <random>

#include "Dense.h"
#include "activation/SoftmaxActivation.h"
#include "loss/Loss.h"
#include "loss/CategoricalCrossEntropyLoss.h"
#include "optimizer/Optimizer.h"
#include "math/Mat.h"
#include "math/Vec.h"

class Network {
private:
	std::vector<Dense*> layers;
	bool accuracyEnabled = false;
	std::function<bool(const Mat<float> &prediction, const Mat<float> &target)> accuracyFunc;

public:
	Network();
	Network(std::initializer_list<Dense*> layers);

	void append(Dense *layer);

	//Evaluate the prediction for an input batch
	//input rows: input size, cols: batch size
	Mat<float> evaluate(const Mat<float> &input);

	//Get average loss for input batch
	//input rows: input size, cols: batch size
	//target rows: target size, cols: batch size
	float getLoss(
		const Mat<float> &input, const Mat<float> &target, const Loss *loss
	);

	void setAccuracyFunc(std::function<bool(const Mat<float> &prediction, const Mat<float> &target)> accuracyFunc);

	void train(
		const Vec<float> *testInputs, const Vec<float> *testTargets, size_t testSize,
		const Loss *loss, Optimizer *optimizer, unsigned int epochs,
		size_t batchSize = 32, bool shuffle = true,
		const Vec<float> *valInputs = nullptr, const Vec<float> *valTargets = nullptr, size_t valSize = 0
	);

	const Dense *getLayer(size_t index) const;

	void display() const;
};

#include "../source/Network.inl"

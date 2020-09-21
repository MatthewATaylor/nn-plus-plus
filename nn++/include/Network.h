#pragma once

#include <vector>
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

	void train(
		const Vec<float> *inputs, const Vec<float> *targets, size_t dataSize,
		const Loss *loss, Optimizer *optimizer, unsigned int epochs,
		size_t batchSize = 32, bool shuffle = true
	);

	void display() const;
};

#include "../source/Network.inl"

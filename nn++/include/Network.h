#pragma once

#include <vector>
#include <initializer_list>
#include <typeinfo>
#include <stdexcept>
#include <iostream>

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

	Vec<float> evaluate(const Vec<float> &input);
	float getLoss(
		const Vec<float> &input, const Vec<float> &target, const Loss *loss
	);
	void train(
		const Vec<float> *inputs, const Vec<float> *targets, size_t dataSize,
		const Loss *loss, Optimizer *optimizer, unsigned int epochs
	);

	void display() const;
};

#include "../source/Network.inl"

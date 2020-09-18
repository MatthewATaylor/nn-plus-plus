#pragma once

#include <cmath>
#include <random>

#include "activation/Activation.h"
#include "math/Mat.h"

class Dense {
private:
	const Activation *activation = nullptr;

public:
	const size_t inputSize;
	const size_t units;

	Mat<float> weights; //rows: units, cols: input size

	Vec<float> biases; //size: units
	
	Vec<float> weightedInputs; //size: units
	Vec<float> activations; //size: units

	Vec<float> errors; //size: units

	//For momentum optimizer
	Mat<float> weightVelocity; //rows: units, cols: input size
	Vec<float> biasVelocity; //size: units

	Dense(size_t inputSize, size_t units, const Activation *activation);

	//Sets layer weighted inputs and activations
	void evaluate(const Vec<float> &input);

	//Returns derivative of activation function with respect to weighted inputs
	Vec<float> activationFuncDerivative() const;

	const Activation *getActivation() const;
};

#include "../source/Dense.inl"

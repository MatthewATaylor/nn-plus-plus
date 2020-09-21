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
	
	Mat<float> weightedInputs; //rows: units, cols: batch size
	Mat<float> activations; //rows: units, cols: batch size

	Mat<float> errors; //rows: units, cols: batch size

	//For momentum
	Mat<float> weightV; //rows: units, cols: input size
	Vec<float> biasV; //size: units

	//For RMSProp
	Mat<float> weightS; //rows: units, cols: input size
	Vec<float> biasS; //size: units

	Dense(size_t inputSize, size_t units, const Activation *activation);

	//Sets layer weighted inputs and activations
	//input rows: input size, cols: batch size
	void evaluate(const Mat<float> &input);

	//Returns derivative of activation function with respect to weighted inputs
	//output rows: units, cols: batch size
	Mat<float> activationFuncDerivative() const;

	const Activation *getActivation() const;
};

#include "../source/Dense.inl"

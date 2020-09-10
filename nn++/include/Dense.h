#pragma once

#include <random>

#include "Activation.h"
#include "math/Mat.h"

template <size_t INPUT_SIZE, size_t UNITS, typename ActivationType>
class Dense {
public:
	Mat<float, UNITS, INPUT_SIZE> weights;
	Vec<float, UNITS> biases;
	
	Vec<float, UNITS> weightedInputs;
	Vec<float, UNITS> activations;

	Vec<float, UNITS> errors;

	Dense();

	//Sets layer weighted inputs and activations
	void evaluate(const Vec<float, INPUT_SIZE> &input);

	//Returns derivative of activation function with respect to weighted inputs
	Vec<float, UNITS> activationFuncDerivative();
};

#include "../source/Dense.inl"

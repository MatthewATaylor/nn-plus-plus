#pragma once

#include <random>

#include "Activation.h"
#include "math/Mat.h"

template <size_t INPUT_SIZE, size_t UNITS, typename ActivationType>
class Dense {
public:
	Mat<float, UNITS, INPUT_SIZE> weights;
	Vec<float, UNITS> biases;

	Dense();
	Vec<float, UNITS> evaluate(const Vec<float, INPUT_SIZE> &input) const;
};

#include "../source/Dense.inl"

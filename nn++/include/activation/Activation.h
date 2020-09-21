#pragma once

#include "math/Mat.h"

class Activation {
protected:
	//Used for random weight initialization
	float weightFactor = 1.0f;

public:
	virtual Mat<float> func(const Mat<float> &input) const = 0;
	virtual Mat<float> derivative(const Mat<float> &input) const = 0;

	float getWeightFactor() const;
};

#include "../source/activation/Activation.inl"

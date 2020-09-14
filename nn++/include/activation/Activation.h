#pragma once

#include "math/Vec.h"

class Activation {
protected:
	float weightFactor = 1.0f;

public:
	virtual Vec<float> func(const Vec<float> &input) const = 0;
	virtual Vec<float> derivative(const Vec<float> &input) const = 0;

	float getWeightFactor() const;
};

#include "../source/activation/Activation.inl"

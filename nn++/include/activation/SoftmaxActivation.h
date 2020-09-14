#pragma once

#include <cmath>
#include <stdexcept>

#include "Activation.h"
#include "math/Vec.h"

class SoftmaxActivation : public Activation {
public:
	Vec<float> func(const Vec<float> &input) const override;
	Vec<float> derivative(const Vec<float> &input) const override;
};

#include "../source/activation/SoftmaxActivation.inl"

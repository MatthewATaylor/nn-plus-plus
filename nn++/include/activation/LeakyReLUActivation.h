#pragma once

#include "Activation.h"
#include "math/Vec.h"

class LeakyReLUActivation : public Activation {
public:
	Vec<float> func(const Vec<float> &input) const override;
	Vec<float> derivative(const Vec<float> &input) const override;
};

#include "../source/activation/LeakyReLUActivation.inl"

#pragma once

#include "Activation.h"
#include "math/Mat.h"

class LeakyReLUActivation : public Activation {
private:
	float leakSlope;

public:
	LeakyReLUActivation(float leakSlope = 0.01f);

	Mat<float> func(const Mat<float> &input) const override;
	Mat<float> derivative(const Mat<float> &input) const override;
};

#include "../source/activation/LeakyReLUActivation.inl"

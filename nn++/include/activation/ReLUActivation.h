#pragma once

#include "Activation.h"
#include "math/Mat.h"

class ReLUActivation : public Activation {
public:
	ReLUActivation();

	Mat<float> func(const Mat<float> &input) const override;
	Mat<float> derivative(const Mat<float> &input) const override;
};

#include "../source/activation/ReLUActivation.inl"

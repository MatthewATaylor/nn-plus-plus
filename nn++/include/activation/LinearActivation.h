#pragma once

#include "Activation.h"
#include "math/Mat.h"

class LinearActivation : public Activation {
public:
	Mat<float> func(const Mat<float> &input) const override;
	Mat<float> derivative(const Mat<float> &input) const override;
};

#include "../source/activation/LinearActivation.inl"

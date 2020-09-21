#pragma once

#include <cmath>

#include "Activation.h"
#include "math/Mat.h"

class TanhActivation : public Activation {
public:
	Mat<float> func(const Mat<float> &input) const override;
	Mat<float> derivative(const Mat<float> &input) const override;
};

#include "../source/activation/TanhActivation.inl"

#pragma once

#include <cmath>
#include <stdexcept>

#include "Activation.h"
#include "math/Mat.h"
#include "math/Vec.h"

class SoftmaxActivation : public Activation {
public:
	Mat<float> func(const Mat<float> &input) const override;
	Mat<float> derivative(const Mat<float> &input) const override;
};

#include "../source/activation/SoftmaxActivation.inl"

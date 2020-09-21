#pragma once

#include <cmath>

#include "loss/Loss.h"
#include "math/Mat.h"

class BinaryCrossEntropyLoss : public Loss {
public:
	float func(
		const Mat<float> &actual, const Mat<float> &prediction
	) const override;

	Mat<float> derivative(
		const Mat<float> &actual, const Mat<float> &prediction
	) const override;
};

#include "../source/loss/BinaryCrossEntropyLoss.inl"

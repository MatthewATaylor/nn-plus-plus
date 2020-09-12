#pragma once

#include "loss/Loss.h"
#include "math/Vec.h"

class MeanSquaredErrorLoss : public Loss {
public:
	float func(
		const Vec<float> &actual, const Vec<float> &prediction
	) const override;

	Vec<float> derivative(
		const Vec<float> &actual, const Vec<float> &prediction
	) const override;
};

#include "../source/loss/MeanSquaredErrorLoss.inl"

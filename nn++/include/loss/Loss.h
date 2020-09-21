#pragma once

#include "math/Mat.h"

class Loss {
public:
	virtual float func(
		const Mat<float> &actual, const Mat<float> &prediction
	) const = 0;

	virtual Mat<float> derivative(
		const Mat<float> &actual, const Mat<float> &prediction
	) const = 0;
};

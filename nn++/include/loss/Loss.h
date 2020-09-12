#pragma once

#include "math/Vec.h"

class Loss {
public:
	virtual float func(
		const Vec<float> &actual, const Vec<float> &prediction
	) const = 0;

	virtual Vec<float> derivative(
		const Vec<float> &actual, const Vec<float> &prediction
	) const = 0;
};

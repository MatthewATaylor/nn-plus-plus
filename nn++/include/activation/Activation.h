#pragma once

#include "math/Vec.h"

class Activation {
public:
	virtual Vec<float> func(const Vec<float> &input) const = 0;
	virtual Vec<float> derivative(const Vec<float> &input) const = 0;
};

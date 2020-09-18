#pragma once

#include "Dense.h"
#include "math/Vec.h"

class Optimizer {
public:
	virtual void updateLayer(Dense *layer, const Vec<float> &prevNodes) = 0;
};

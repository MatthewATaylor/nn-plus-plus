#pragma once

#include "Dense.h"
#include "math/Mat.h"

class Optimizer {
public:
	virtual void updateLayer(Dense *layer, const Mat<float> &prevNodes, unsigned int timestep) = 0;
};

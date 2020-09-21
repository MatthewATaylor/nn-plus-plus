#pragma once

#include <cmath>

#include "Dense.h"
#include "Optimizer.h"
#include "math/Vec.h"
#include "math/Mat.h"

class RMSPropOptimizer : public Optimizer {
private:
	const float LEARNING_RATE;
	const float BETA;

public:
	RMSPropOptimizer(float learningRate = 0.001f, float beta = 0.9f);
	void updateLayer(Dense *layer, const Mat<float> &prevNodes, unsigned int timestep) override;
};

#include "../source/optimizer/RMSPropOptimizer.inl"

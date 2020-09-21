#pragma once

#include <cmath>

#include "Dense.h"
#include "Optimizer.h"
#include "math/Vec.h"
#include "math/Mat.h"

class AdamOptimizer : public Optimizer {
private:
	const float LEARNING_RATE;
	const float BETA_1;
	const float BETA_2;

public:
	AdamOptimizer(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f);
	void updateLayer(Dense *layer, const Mat<float> &prevNodes, unsigned int timestep) override;
};

#include "../source/optimizer/AdamOptimizer.inl"

#pragma once

#include "Dense.h"
#include "Optimizer.h"
#include "math/Vec.h"
#include "math/Mat.h"

class MomentumOptimizer : public Optimizer {
private:
	const float LEARNING_RATE;
	const float BETA;

public:
	MomentumOptimizer(float learningRate = 0.001f, float beta = 0.9f);
	void updateLayer(Dense *layer, const Mat<float> &prevNodes, unsigned int timestep) override;
};

#include "../source/optimizer/MomentumOptimizer.inl"

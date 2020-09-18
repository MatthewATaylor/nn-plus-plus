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
	MomentumOptimizer(float learningRate, float beta);
	void updateLayer(Dense *layer, const Vec<float> &prevNodes) override;
};

#include "../source/optimizer/MomentumOptimizer.inl"

#pragma once

#include "Dense.h"
#include "Optimizer.h"
#include "math/Vec.h"
#include "math/Mat.h"

class SGDOptimizer : public Optimizer {
private:
	const float LEARNING_RATE;

public:
	SGDOptimizer(float learningRate);
	void updateLayer(Dense *layer, const Mat<float> &prevNodes, unsigned int timestep) override;
};

#include "../source/optimizer/SGDOptimizer.inl"

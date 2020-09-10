#pragma once

#include <array>
#include <tuple>
#include <iostream>

#include "Dense.h"
#include "math/Mat.h"
#include "math/Vec.h"

template <typename... LayerTypes>
class Network {
private:
	std::tuple<LayerTypes*...> layers;

	template <
		size_t INPUT_SIZE, size_t OUTPUT_SIZE,
		typename NextLayerType
	>
	static Vec<float, OUTPUT_SIZE> evaluate(
		const Vec<float, INPUT_SIZE> &input,
		NextLayerType *layer
	);

	template <
		size_t INPUT_SIZE, size_t OUTPUT_SIZE,
		typename NextLayerType, typename... OtherLayerTypes
	>
	static Vec<float, OUTPUT_SIZE> evaluate(
		const Vec<float, INPUT_SIZE> &input,
		NextLayerType *layer,
		OtherLayerTypes*... otherLayers
	);

	template <size_t LAYER_NUM>
	void backpropagateLayer();

	template <
		size_t INPUT_SIZE, size_t OUTPUT_SIZE, size_t BATCH_SIZE,
		typename LossType
	>
	void trainStep(
		const std::array<Vec<float, INPUT_SIZE>, BATCH_SIZE> &x,
		const std::array<Vec<float, OUTPUT_SIZE>, BATCH_SIZE> &y
	);

public:
	Network(LayerTypes*... layers);

	template <size_t INPUT_SIZE, size_t OUTPUT_SIZE, typename LossType>
	float getLoss(
		const Vec<float, INPUT_SIZE> &input,
		const Vec<float, OUTPUT_SIZE> &target
	);

	template <
		size_t NUM_INPUTS,
		size_t INPUT_SIZE, size_t OUTPUT_SIZE, size_t BATCH_SIZE,
		typename LossType
	>
	void train(
		const std::array<Vec<float, INPUT_SIZE>, NUM_INPUTS> &x,
		const std::array<Vec<float, OUTPUT_SIZE>, NUM_INPUTS> &y
	);
};

#include "../source/Network.inl"

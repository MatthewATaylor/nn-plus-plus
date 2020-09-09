#pragma once

#include <tuple>

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
		const NextLayerType *layer
	);

	template <
		size_t INPUT_SIZE, size_t OUTPUT_SIZE,
		typename NextLayerType, typename... OtherLayerTypes
	>
	static Vec<float, OUTPUT_SIZE> evaluate(
		const Vec<float, INPUT_SIZE> &input,
		const NextLayerType *layer,
		const OtherLayerTypes*... otherLayers
	);

public:
	Network(LayerTypes*... layers);

	template <size_t INPUT_SIZE, size_t OUTPUT_SIZE, typename LossType>
	float getLoss(
		const Vec<float, INPUT_SIZE> &input,
		const Vec<float, OUTPUT_SIZE> &target
	);
};

#include "../source/Network.inl"

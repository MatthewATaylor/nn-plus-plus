#pragma once

template <typename... LayerTypes>
template <
	size_t INPUT_SIZE, size_t OUTPUT_SIZE,
	typename NextLayerType
>
inline Vec<float, OUTPUT_SIZE> Network<LayerTypes...>::evaluate(
	const Vec<float, INPUT_SIZE> &input,
	NextLayerType *layer
) {
	layer->evaluate(input);
	return layer->activations;
}

template <typename... LayerTypes>
template <
	size_t INPUT_SIZE, size_t OUTPUT_SIZE,
	typename NextLayerType, typename... OtherLayerTypes
>
inline Vec<float, OUTPUT_SIZE> Network<LayerTypes...>::evaluate(
	const Vec<float, INPUT_SIZE> &input,
	NextLayerType *layer,
	OtherLayerTypes*... otherLayers
) {
	layer->evaluate(input);
	Vec<float, decltype(layer->weights)::rows> output = layer->activations;
	return evaluate<decltype(layer->weights)::rows, OUTPUT_SIZE, OtherLayerTypes...>(
		output, otherLayers...
	);
}

template <typename... LayerTypes>
template <size_t LAYER_NUM>
inline void Network<LayerTypes...>::backpropagateLayer() {
	if constexpr (LAYER_NUM <= 0) {
		return;
	}
	
	auto layer = std::get<LAYER_NUM>(layers);
	auto nextLayer = std::get<LAYER_NUM>(layers);
	auto weights = layer->weights.transpose();
	auto errors = layer->errors;
	auto nextErrors = weights * errors;
	auto nextActivationFuncDerivative = nextLayer->activationFuncDerivative();

	//Compute next layer's errors
	//for (size_t i = 0; i < nextErrors::size; ++i) {
	//	nextErrors.set(i, nextErrors.get(i) * nextActivationFuncDerivative.get(i));
	//}

	backpropagateLayer<LAYER_NUM - 1>();
}

template <typename... LayerTypes>
template <
	size_t INPUT_SIZE, size_t OUTPUT_SIZE, size_t BATCH_SIZE,
	typename LossType
>
inline void Network<LayerTypes...>::trainStep(
	const std::array<Vec<float, INPUT_SIZE>, BATCH_SIZE> &x,
	const std::array<Vec<float, OUTPUT_SIZE>, BATCH_SIZE> &y
) {

}

template <typename... LayerTypes>
inline Network<LayerTypes...>::Network(LayerTypes*... layers) {
	this->layers = std::tuple<LayerTypes*...>(layers...);
}

template <typename... LayerTypes>
template <size_t INPUT_SIZE, size_t OUTPUT_SIZE, typename LossType>
inline float Network<LayerTypes...>::getLoss(
	const Vec<float, INPUT_SIZE> &input,
	const Vec<float, OUTPUT_SIZE> &target
) {
	Vec<float, OUTPUT_SIZE> prediction = std::apply(
		evaluate<INPUT_SIZE, OUTPUT_SIZE, LayerTypes...>,
		std::tuple_cat(std::make_tuple(input), layers)
	);
	return LossType::func(target, prediction);
}

template <typename... LayerTypes>
template <
	size_t NUM_INPUTS,
	size_t INPUT_SIZE, size_t OUTPUT_SIZE, size_t BATCH_SIZE,
	typename LossType
>
inline void Network<LayerTypes...>::train(
	const std::array<Vec<float, INPUT_SIZE>, NUM_INPUTS> &x,
	const std::array<Vec<float, OUTPUT_SIZE>, NUM_INPUTS> &y
) {
	for (size_t i = 0; i < NUM_INPUTS; ++i) {
		Vec<float, OUTPUT_SIZE> prediction = std::apply(
			evaluate<INPUT_SIZE, OUTPUT_SIZE, LayerTypes...>,
			std::tuple_cat(std::make_tuple(x[i]), layers)
		);

		//Compute error (dC/dz) of output layer
		constexpr size_t numLayers = sizeof...(LayerTypes);
		auto lastLayer = std::get<numLayers - 1>(layers);
		lastLayer->errors = LossType::derivative(y[i], prediction);
		Vec<float, OUTPUT_SIZE> outputActivationFuncDerivative = lastLayer->activationFuncDerivative();
		for (size_t j = 0; j < OUTPUT_SIZE; ++j) {
			lastLayer->errors.set(j, lastLayer->errors.get(j) * outputActivationFuncDerivative.get(j));
		}

		backpropagateLayer<numLayers - 1>();
	}
}

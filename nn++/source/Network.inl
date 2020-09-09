#pragma once

template <typename... LayerTypes>
template <
	size_t INPUT_SIZE, size_t OUTPUT_SIZE,
	typename NextLayerType
>
Vec<float, OUTPUT_SIZE> Network<LayerTypes...>::evaluate(
	const Vec<float, INPUT_SIZE> &input,
	const NextLayerType *layer
) {
	return layer->evaluate(input);
}

template <typename... LayerTypes>
template <
	size_t INPUT_SIZE, size_t OUTPUT_SIZE,
	typename NextLayerType, typename... OtherLayerTypes
>
Vec<float, OUTPUT_SIZE> Network<LayerTypes...>::evaluate(
	const Vec<float, INPUT_SIZE> &input,
	const NextLayerType *layer,
	const OtherLayerTypes*... otherLayers
) {
	Vec<float, decltype(layer->weights)::rows> output = layer->evaluate(input);
	return evaluate<decltype(layer->weights)::rows, OUTPUT_SIZE, OtherLayerTypes...>(
		output, otherLayers...
	);
}

template <typename... LayerTypes>
Network<LayerTypes...>::Network(LayerTypes*... layers) {
	this->layers = std::tuple<LayerTypes*...>(layers...);
}

template <typename... LayerTypes>
template <size_t INPUT_SIZE, size_t OUTPUT_SIZE, typename LossType>
float Network<LayerTypes...>::getLoss(
	const Vec<float, INPUT_SIZE> &input,
	const Vec<float, OUTPUT_SIZE> &target
) {
	Vec<float, OUTPUT_SIZE> prediction = std::apply(
		evaluate<INPUT_SIZE, OUTPUT_SIZE, LayerTypes...>,
		std::tuple_cat(std::make_tuple(input), layers)
	);
	return LossType::func(target, prediction);
}

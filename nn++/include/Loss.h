#pragma once

#include <cmath>

#include "math/Vec.h"

class Loss {
public:
	class MeanSquaredError {
	public:
		template <size_t OUTPUT_SIZE>
		static float func(
			const Vec<float, OUTPUT_SIZE> &actual,
			const Vec<float, OUTPUT_SIZE> &prediction
		);

		template <size_t OUTPUT_SIZE>
		static Vec<float, OUTPUT_SIZE> derivative(
			const Vec<float, OUTPUT_SIZE> &actual,
			const Vec<float, OUTPUT_SIZE> &prediction
		);
	};

	class BinaryCrossEntropy {
	public:
		template <size_t OUTPUT_SIZE>
		static float func(
			const Vec<float, OUTPUT_SIZE> &actual,
			const Vec<float, OUTPUT_SIZE> &prediction
		);

		template <size_t OUTPUT_SIZE>
		static Vec<float, OUTPUT_SIZE> derivative(
			const Vec<float, OUTPUT_SIZE> &actual,
			const Vec<float, OUTPUT_SIZE> &prediction
		);
	};

	class CategoricalCrossEntropy {
	public:
		template <size_t OUTPUT_SIZE>
		static float func(
			const Vec<float, OUTPUT_SIZE> &actual,
			const Vec<float, OUTPUT_SIZE> &prediction
		);

		template <size_t OUTPUT_SIZE>
		static Vec<float, OUTPUT_SIZE> derivative(
			const Vec<float, OUTPUT_SIZE> &actual,
			const Vec<float, OUTPUT_SIZE> &prediction
		);
	};
};

#include "../source/Loss.inl"

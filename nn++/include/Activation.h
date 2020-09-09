#pragma once

#include <cmath>

#include "math/Vec.h"

class Activation {
public:
	class ReLU {
	public:
		template <size_t UNITS>
		static Vec<float, UNITS> func(const Vec<float, UNITS> &input);

		template <size_t UNITS>
		static Vec<float, UNITS> derivative(const Vec<float, UNITS> &input);
	};

	class LeakyReLU {
	public:
		template <size_t UNITS>
		static Vec<float, UNITS> func(const Vec<float, UNITS> &input);

		template <size_t UNITS>
		static Vec<float, UNITS> derivative(const Vec<float, UNITS> &input);
	};

	class Linear {
	public:
		template <size_t UNITS>
		static Vec<float, UNITS> func(const Vec<float, UNITS> &input);

		template <size_t UNITS>
		static Vec<float, UNITS> derivative(const Vec<float, UNITS> &input);
	};

	class Sigmoid {
	public:
		template <size_t UNITS>
		static Vec<float, UNITS> func(const Vec<float, UNITS> &input);

		template <size_t UNITS>
		static Vec<float, UNITS> derivative(const Vec<float, UNITS> &input);
	};

	class Tanh {
	public:
		template <size_t UNITS>
		static Vec<float, UNITS> func(const Vec<float, UNITS> &input);

		template <size_t UNITS>
		static Vec<float, UNITS> derivative(const Vec<float, UNITS> &input);
	};
};

#include "../source/Activation.inl"

#pragma once

#include <cmath>
#include <ostream>
#include <initializer_list>

#include "Vec.h"

template <typename T>
class Mat;

template<typename T>
std::ostream &operator<<(std::ostream &outputStream, const Mat<T> &mat);

template <typename T>
class Mat {
private:
	void reset();

public:
	T **elements = nullptr;

	size_t rows = 0;
	size_t cols = 0;

	Mat();
	Mat(size_t rows, size_t cols);
	Mat(size_t rows, size_t cols, T element);
	Mat(std::initializer_list<std::initializer_list<T>> elements);
	Mat(const Vec<T> &vec, bool colVec = true);
	Mat(const Mat<T> &mat);
	Mat(Mat<T> &&mat);

	~Mat();

	T &operator()(size_t row, size_t col);
	const T &operator()(size_t row, size_t col) const;

	Mat<T> &operator=(const Mat<T> &otherMat);
	Mat<T> &operator=(Mat<T> &&otherMat);
	Mat<T> &operator+=(const Mat<T> &otherMat);
	Mat<T> &operator-=(const Mat<T> &otherMat);

	Mat<T> &operator+=(T value);
	Mat<T> &operator-=(T value);
	Mat<T> &operator*=(T value);
	Mat<T> &operator/=(T value);

	Mat<T> operator+(const Mat<T> &otherMat) const;
	Mat<T> operator-(const Mat<T> &otherMat) const;
	Mat<T> operator*(const Mat<T> &otherMat) const;
	Vec<T> operator*(const Vec<T> &vec) const;

	Mat<T> operator+(T value);
	Mat<T> operator-(T value);
	Mat<T> operator*(T value);
	Mat<T> operator/(T value);

	friend std::ostream &operator<<<T>(std::ostream &outputStream, const Mat<T> &mat);

	Mat<T> transpose() const;

	template <typename U>
	Mat<T> powByElements(U power) const;
	
	Mat<T> divideByElements(const Mat<T> &otherMat) const;
};

#include "../../source/math/Mat.inl"

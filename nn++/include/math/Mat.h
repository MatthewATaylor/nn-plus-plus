#pragma once

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
	Mat(const Mat<T> &mat);
	Mat(Mat<T> &&mat);

	~Mat();

	T &operator()(size_t row, size_t col);
	const T &operator()(size_t row, size_t col) const;

	Mat<T> &operator=(const Mat<T> &otherMat);
	Mat<T> &operator=(Mat<T> &&otherMat);
	Mat<T> &operator+=(const Mat<T> &otherMat);
	Mat<T> &operator-=(const Mat<T> &otherMat);

	Mat<T> operator+(const Mat<T> &otherMat) const;
	Mat<T> operator-(const Mat<T> &otherMat) const;
	Mat<T> operator*(const Mat<T> &otherMat) const;
	Vec<T> operator*(const Vec<T> &vec) const;

	friend std::ostream &operator<<<T>(std::ostream &outputStream, const Mat<T> &mat);

	Mat<T> transpose() const;
};

#include "../../source/math/Mat.inl"

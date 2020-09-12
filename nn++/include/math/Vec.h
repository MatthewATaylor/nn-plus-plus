#pragma once

#include <cmath>
#include <ostream>
#include <initializer_list>

template<typename T>
class Vec;

template<typename T>
std::ostream &operator<<(std::ostream &outputStream, const Vec<T> &vec);

template<typename T>
class Vec {
private:
	void reset();

public:
	T *elements = nullptr;
	size_t size = 0;

	Vec();
	Vec(size_t size);
	Vec(size_t size, T element);
	Vec(std::initializer_list<T> elements);
	Vec(const Vec<T> &vec);
	Vec(Vec<T> &&vec);

	~Vec();

	T &operator()(size_t index);
	const T &operator()(size_t index) const;

	Vec<T> &operator=(const Vec<T> &otherVec);
	Vec<T> &operator=(Vec<T> &&otherVec);
	Vec<T> &operator+=(const Vec<T> &otherVec);
	Vec<T> &operator-=(const Vec<T> &otherVec);
	Vec<T> &operator*=(const Vec<T> &otherVec);
	Vec<T> &operator/=(const Vec<T> &otherVec);

	Vec<T> &operator+=(T value);
	Vec<T> &operator-=(T value);
	Vec<T> &operator*=(T value);
	Vec<T> &operator/=(T value);

	Vec<T> operator+(const Vec<T> &otherVec) const;
	Vec<T> operator-(const Vec<T> &otherVec) const;
	Vec<T> operator*(const Vec<T> &otherVec) const;
	Vec<T> operator/(const Vec<T> &otherVec) const;

	Vec<T> operator+(T value) const;
	Vec<T> operator-(T value) const;
	Vec<T> operator*(T value) const;
	Vec<T> operator/(T value) const;

	bool operator==(const Vec<T> &otherVec) const;

	friend std::ostream &operator<<<T>(std::ostream &outputStream, const Vec<T> &vec);

	T dot(const Vec<T> &otherVec) const;
	T magSquared() const;
	T mag() const;
};

#include "../../source/math/Vec.inl"

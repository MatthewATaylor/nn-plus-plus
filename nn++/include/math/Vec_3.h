#pragma once

template<typename T>
class Vec<T, 3>;

template<typename T>
std::ostream &operator<<(std::ostream &outputStream, const Vec<T, 3> &vec);

template<typename T>
class Vec<T, 3> {
private:
	T elements[3];

public:
	static constexpr size_t size = 3;

	Vec();
	Vec(T elements[3]);
	Vec(T x, T y, T z);
	Vec(T element);

	Vec<T, 3> &operator=(const Vec<T, 3> &otherVec);
	Vec<T, 3> &operator+=(const Vec<T, 3> &otherVec);
	Vec<T, 3> &operator-=(const Vec<T, 3> &otherVec);

	Vec<T, 3> &operator+=(const T value);
	Vec<T, 3> &operator-=(const T value);
	Vec<T, 3> &operator*=(const T value);
	Vec<T, 3> &operator/=(const T value);

	Vec<T, 3> operator+(const Vec<T, 3> &otherVec) const;
	Vec<T, 3> operator-(const Vec<T, 3> &otherVec) const;

	Vec<T, 3> operator+(const T value) const;
	Vec<T, 3> operator-(const T value) const;
	Vec<T, 3> operator*(const T value) const;
	Vec<T, 3> operator/(const T value) const;

	bool operator==(const Vec<T, 3> &otherVec) const;

	friend std::ostream &operator<<<T, 3>(std::ostream &outputStream, const Vec<T, 3> &vec);

	T dot(const Vec<T, 3> &otherVec) const;
	Vec<T, 3> cross(const Vec<T, 3> &otherVec) const;
	T magSquared() const;
	T mag() const;

	T get(size_t index) const;
	T getX() const;
	T getY() const;
	T getZ() const;
	void set(size_t index, T newElement);
	void setX(T newElement);
	void setY(T newElement);
	void setZ(T newElement);
};

#include "../../source/math/Vec_3.inl"

#pragma once

template<typename T>
inline Vec<T, 3>::Vec() {
	for (size_t i = 0; i < 3; ++i) {
		elements[i] = 0;
	}
}
template<typename T>
inline Vec<T, 3>::Vec(T elements[3]) {
	for (size_t i = 0; i < 3; ++i) {
		this->elements[i] = elements[i];
	}
}
template<typename T>
inline Vec<T, 3>::Vec(T x, T y, T z) {
	elements[0] = x;
	elements[1] = y;
	elements[2] = z;
}
template<typename T>
inline Vec<T, 3>::Vec(T element) {
	for (size_t i = 0; i < 3; ++i) {
		this->elements[i] = element;
	}
}

template<typename T>
inline Vec<T, 3> &Vec<T, 3>::operator=(const Vec<T, 3> &otherVec) {
	for (size_t i = 0; i < 3; ++i) {
		elements[i] = otherVec.get(i);
	}
	return *this;
}
template<typename T>
inline Vec<T, 3> &Vec<T, 3>::operator+=(const Vec<T, 3> &otherVec) {
	for (size_t i = 0; i < 3; ++i) {
		elements[i] += otherVec.get(i);
	}
	return *this;
}
template<typename T>
inline Vec<T, 3> &Vec<T, 3>::operator-=(const Vec<T, 3> &otherVec) {
	for (size_t i = 0; i < 3; ++i) {
		elements[i] -= otherVec.get(i);
	}
	return *this;
}

template<typename T>
inline Vec<T, 3> &Vec<T, 3>::operator+=(const T value) {
	for (size_t i = 0; i < 3; ++i) {
		elements[i] += value;
	}
	return *this;
}
template<typename T>
inline Vec<T, 3> &Vec<T, 3>::operator-=(const T value) {
	for (size_t i = 0; i < 3; ++i) {
		elements[i] -= value;
	}
	return *this;
}
template<typename T>
inline Vec<T, 3> &Vec<T, 3>::operator*=(const T value) {
	for (size_t i = 0; i < 3; ++i) {
		elements[i] *= value;
	}
	return *this;
}
template<typename T>
inline Vec<T, 3> &Vec<T, 3>::operator/=(const T value) {
	for (size_t i = 0; i < 3; ++i) {
		elements[i] /= value;
	}
	return *this;
}

template<typename T>
inline Vec<T, 3> Vec<T, 3>::operator+(const Vec<T, 3> &otherVec) const {
	Vec<T, 3> vectorResult = *this;
	return vectorResult += otherVec;
}
template<typename T>
inline Vec<T, 3> Vec<T, 3>::operator-(const Vec<T, 3> &otherVec) const {
	Vec<T, 3> vectorResult = *this;
	return vectorResult -= otherVec;
}

template<typename T>
inline Vec<T, 3> Vec<T, 3>::operator+(const T value) const {
	Vec<T, 3> vectorResult = *this;
	return vectorResult += value;
}
template<typename T>
inline Vec<T, 3> Vec<T, 3>::operator-(const T value) const {
	Vec<T, 3> vectorResult = *this;
	return vectorResult -= value;
}
template<typename T>
inline Vec<T, 3> Vec<T, 3>::operator*(const T value) const {
	Vec<T, 3> vectorResult = *this;
	return vectorResult *= value;
}
template<typename T>
inline Vec<T, 3> Vec<T, 3>::operator/(const T value) const {
	Vec<T, 3> vectorResult = *this;
	return vectorResult /= value;
}

template<typename T>
inline bool Vec<T, 3>::operator==(const Vec<T, 3> &otherVec) const {
	for (size_t i = 0; i < 3; ++i) {
		if (elements[i] != otherVec.get(i)) {
			return false;
		}
	}
	return true;
}

template<typename T>
inline std::ostream &operator<<(std::ostream &outputStream, const Vec<T, 3> &vec) {
	outputStream << "<";
	for (size_t i = 0; i < 3; ++i) {
		outputStream << vec.get(i);
		if (i < 3 - 1) {
			outputStream << ", ";
		}
	}
	outputStream << ">";
	return outputStream;
}

template<typename T>
inline T Vec<T, 3>::dot(const Vec<T, 3> &otherVec) const {
	T result = 0;
	for (size_t i = 0; i < 3; ++i) {
		result += elements[i] * otherVec.get(i);
	}
	return result;
}
template<typename T>
inline Vec<T, 3> Vec<T, 3>::cross(const Vec<T, 3> &otherVec) const {
	T newX = getY() * otherVec.getZ() - getZ() * otherVec.getY();
	T newY = -1 * (getX() * otherVec.getZ() - getZ() * otherVec.getX());
	T newZ = getX() * otherVec.getY() - getY() * otherVec.getX();
	Vec<T, 3> newVec(newX, newY, newZ);
	return newVec;
}
template<typename T>
inline T Vec<T, 3>::magSquared() const {
	T result = 0;
	for (size_t i = 0; i < 3; ++i) {
		result += elements[i] * elements[i];
	}
	return result;
}
template<typename T>
inline T Vec<T, 3>::mag() const {
	return sqrt(magSquared());
}

template<typename T>
inline T Vec<T, 3>::get(size_t index) const {
	return elements[index];
}
template<typename T>
inline T Vec<T, 3>::getX() const {
	return elements[0];
}
template<typename T>
inline T Vec<T, 3>::getY() const {
	return elements[1];
}
template<typename T>
inline T Vec<T, 3>::getZ() const {
	return elements[2];
}
template<typename T>
inline void Vec<T, 3>::set(size_t index, T newElement) {
	elements[index] = newElement;
}
template<typename T>
inline void Vec<T, 3>::setX(T newElement) {
	elements[0] = newElement;
}
template<typename T>
inline void Vec<T, 3>::setY(T newElement) {
	elements[1] = newElement;
}
template<typename T>
inline void Vec<T, 3>::setZ(T newElement) {
	elements[2] = newElement;
}

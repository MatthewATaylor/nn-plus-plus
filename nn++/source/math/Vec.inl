#pragma once

template <typename T>
void Vec<T>::reset() {
	delete[] elements;
}

template <typename T>
inline Vec<T>::Vec() {}
template <typename T>
inline Vec<T>::Vec(size_t size) {
	this->size = size;
	elements = new T[size];

	for (size_t i = 0; i < size; ++i) {
		elements[i] = 0;
	}
}
template <typename T>
inline Vec<T>::Vec(size_t size, T element) {
	this->size = size;
	elements = new T[size];

	for (size_t i = 0; i < size; ++i) {
		elements[i] = element;
	}
}
template <typename T>
inline Vec<T>::Vec(std::initializer_list<T> elements) {
	this->size = elements.size();
	this->elements = new T[size];

	size_t i = 0;
	for (const auto &element : elements) {
		this->elements[i] = element;
		++i;
	}
}
template <typename T>
inline Vec<T>::Vec(const Vec<T> &vec) {
	size = vec.size;
	elements = new T[size];

	for (size_t i = 0; i < size; ++i) {
		elements[i] = vec(i);
	}
}
template <typename T>
inline Vec<T>::Vec(Vec<T> &&vec) {
	size = vec.size;
	elements = vec.elements;
	vec.elements = nullptr;
}

template <typename T>
inline Vec<T>::~Vec() {
	reset();
}

template <typename T>
inline T &Vec<T>::operator()(size_t index) {
	return elements[index];
}
template <typename T>
inline const T &Vec<T>::operator()(size_t index) const {
	return elements[index];
}

template <typename T>
inline Vec<T> &Vec<T>::operator=(const Vec<T> &otherVec) {
	reset();
	size = otherVec.size;
	elements = new T[size];
	for (size_t i = 0; i < size; ++i) {
		elements[i] = otherVec(i);
	}
}
template <typename T>
inline Vec<T> &Vec<T>::operator=(Vec<T> &&otherVec) {
	reset();
	size = otherVec.size;
	elements = otherVec.elements;
	otherVec.elements = nullptr;
}
template <typename T>
inline Vec<T> &Vec<T>::operator+=(const Vec<T> &otherVec) {
	for (size_t i = 0; i < size; ++i) {
		elements[i] += otherVec(i);
	}
	return *this;
}
template <typename T>
inline Vec<T> &Vec<T>::operator-=(const Vec<T> &otherVec) {
	for (size_t i = 0; i < size; ++i) {
		elements[i] -= otherVec(i);
	}
	return *this;
}
template <typename T>
inline Vec<T> &Vec<T>::operator*=(const Vec<T> &otherVec) {
	for (size_t i = 0; i < size; ++i) {
		elements[i] *= otherVec(i);
	}
	return *this;
}
template <typename T>
inline Vec<T> &Vec<T>::operator/=(const Vec<T> &otherVec) {
	for (size_t i = 0; i < size; ++i) {
		elements[i] /= otherVec(i);
	}
	return *this;
}

template <typename T>
inline Vec<T> &Vec<T>::operator+=(const T value) {
	for (size_t i = 0; i < size; ++i) {
		elements[i] += value;
	}
	return *this;
}
template <typename T>
inline Vec<T> &Vec<T>::operator-=(const T value) {
	for (size_t i = 0; i < size; ++i) {
		elements[i] -= value;
	}
	return *this;
}
template <typename T>
inline Vec<T> &Vec<T>::operator*=(const T value) {
	for (size_t i = 0; i < size; ++i) {
		elements[i] *= value;
	}
	return *this;
}
template <typename T>
inline Vec<T> &Vec<T>::operator/=(const T value) {
	for (size_t i = 0; i < size; ++i) {
		elements[i] /= value;
	}
	return *this;
}

template <typename T>
inline Vec<T> Vec<T>::operator+(const Vec<T> &otherVec) const {
	Vec<T> result = *this;
	return result += otherVec;
}
template <typename T>
inline Vec<T> Vec<T>::operator-(const Vec<T> &otherVec) const {
	Vec<T> result = *this;
	return result -= otherVec;
}
template <typename T>
inline Vec<T> Vec<T>::operator*(const Vec<T> &otherVec) const {
	Vec<T> result = *this;
	return result *= otherVec;
}
template <typename T>
inline Vec<T> Vec<T>::operator/(const Vec<T> &otherVec) const {
	Vec<T> result = *this;
	return result /= otherVec;
}

template <typename T>
inline Vec<T> Vec<T>::operator+(const T value) const {
	Vec<T> result = *this;
	return result += value;
}
template <typename T>
inline Vec<T> Vec<T>::operator-(const T value) const {
	Vec<T> result = *this;
	return result += value;
}
template <typename T>
inline Vec<T> Vec<T>::operator*(const T value) const {
	Vec<T> result = *this;
	return result += value;
}
template <typename T>
inline Vec<T> Vec<T>::operator/(const T value) const {
	Vec<T> result = *this;
	return result += value;
}

template <typename T>
inline bool Vec<T>::operator==(const Vec<T> &otherVec) const {
	for (size_t i = 0; i < size; ++i) {
		if (elements[i] != otherVec(i)) {
			return false;
		}
	}
	return true;
}

template <typename T>
inline std::ostream &operator<<<T>(std::ostream &outputStream, const Vec<T> &vec) {
	outputStream << "<";
	for (size_t i = 0; i < vec.size; ++i) {
		outputStream << vec(i);
		if (i < vec.size - 1) {
			outputStream << ", ";
		}
	}
	outputStream << ">";
	return outputStream;
}

template <typename T>
inline T Vec<T>::dot(const Vec<T> &otherVec) const {
	T result = 0;
	for (size_t i = 0; i < size; ++i) {
		result += elements[i] * otherVec(i);
	}
	return result;
}
template <typename T>
inline T Vec<T>::magSquared() const {
	T result = 0;
	for (size_t i = 0; i < size; ++i) {
		result += elements[i] * elements[i];
	}
	return result;
}
template <typename T>
inline T Vec<T>::mag() const {
	return std::sqrt(magSquared());
}

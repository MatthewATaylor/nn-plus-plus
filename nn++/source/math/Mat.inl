#pragma once

template <typename T>
void Mat<T>::reset() {
	for (size_t i = 0; i < rows; ++i) {
		delete[] elements[i];
	}
	delete[] elements;
}

template <typename T>
inline Mat<T>::Mat() {}
template <typename T>
inline Mat<T>::Mat(size_t rows, size_t cols) {
	this->rows = rows;
	this->cols = cols;

	elements = new T*[rows];
	for (size_t i = 0; i < rows; ++i) {
		elements[i] = new T[cols];
	}
}
template <typename T>
inline Mat<T>::Mat(size_t rows, size_t cols, T element) {
	this->rows = rows;
	this->cols = cols;

	elements = new T*[rows];
	for (size_t i = 0; i < rows; ++i) {
		elements[i] = new T[cols];
		for (size_t j = 0; j < cols; ++j) {
			elements[i][j] = element;
		}
	}
}
template <typename T>
inline Mat<T>::Mat(std::initializer_list<std::initializer_list<T>> elements) {
	this->rows = elements.size();
	this->cols = elements.begin()->size();

	this->elements = new T*[rows];

	size_t rowNum = 0;
	size_t colNum = 0;

	for (const auto &row : elements) {
		colNum = 0;
		this->elements[rowNum] = new T[cols];
		for (const auto &element : row) {
			this->elements[rowNum][colNum] = element;
			++colNum;
		}
		++rowNum;
	}
}
template <typename T>
inline Mat<T>::Mat(const Vec<T> &vec, bool colVec) {
	if (colVec) {
		rows = vec.size;
		cols = 1;
	}
	else {
		rows = 1;
		cols = vec.size;
	}

	elements = new T*[rows];
		
	for (size_t i = 0; i < rows; ++i) {
		elements[i] = new T[cols];
		for (size_t j = 0; j < cols; ++j) {
			elements[i][j] = colVec ? vec(i) : vec(j);
		}
	}
}
template <typename T>
inline Mat<T>::Mat(const Mat<T> &mat) {
	rows = mat.rows;
	cols = mat.cols;

	elements = new T*[rows];

	for (size_t i = 0; i < rows; ++i) {
		elements[i] = new T[cols];
		for (size_t j = 0; j < cols; ++j) {
			elements[i][j] = mat(i, j);
		}
	}
}
template <typename T>
inline Mat<T>::Mat(Mat<T> &&mat) {
	rows = mat.rows;
	cols = mat.cols;
	elements = mat.elements;

	mat.rows = 0;
	mat.cols = 0;
	mat.elements = nullptr;
}

template <typename T>
inline Mat<T>::~Mat() {
	reset();
}

template <typename T>
inline T &Mat<T>::operator()(size_t row, size_t col) {
	return elements[row][col];
}
template <typename T>
inline const T &Mat<T>::operator()(size_t row, size_t col) const {
	return elements[row][col];
}

template <typename T>
inline Mat<T> &Mat<T>::operator=(const Mat<T> &otherMat) {
	reset();

	rows = otherMat.rows;
	cols = otherMat.cols;

	elements = new T*[rows];

	for (size_t i = 0; i < rows; ++i) {
		elements[i] = new T[cols];
		for (size_t j = 0; j < cols; ++j) {
			elements[i][j] = otherMat(i, j);
		}
	}

	return *this;
}
template <typename T>
inline Mat<T> &Mat<T>::operator=(Mat<T> &&otherMat) {
	reset();
	
	rows = otherMat.rows;
	cols = otherMat.cols;
	elements = otherMat.elements;

	otherMat.rows = 0;
	otherMat.cols = 0;
	otherMat.elements = nullptr;

	return *this;
}
template <typename T>
inline Mat<T> &Mat<T>::operator+=(const Mat<T> &otherMat) {
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			elements[i][j] += otherMat(i, j);
		}
	}
	return *this;
}
template <typename T>
inline Mat<T> &Mat<T>::operator-=(const Mat<T> &otherMat) {
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			elements[i][j] -= otherMat(i, j);
		}
	}
	return *this;
}

template <typename T>
inline Mat<T> &Mat<T>::operator+=(T value) {
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			elements[i][j] += value;
		}
	}
	return *this;
}
template <typename T>
inline Mat<T> &Mat<T>::operator-=(T value) {
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			elements[i][j] -= value;
		}
	}
	return *this;
}
template <typename T>
inline Mat<T> &Mat<T>::operator*=(T value) {
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			elements[i][j] *= value;
		}
	}
	return *this;
}
template <typename T>
inline Mat<T> &Mat<T>::operator/=(T value) {
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			elements[i][j] /= value;
		}
	}
	return *this;
}

template <typename T>
inline Mat<T> Mat<T>::operator+(const Mat<T> &otherMat) const {
	Mat<T> result = *this;
	return result += otherMat;
}
template <typename T>
inline Mat<T> Mat<T>::operator-(const Mat<T> &otherMat) const {
	Mat<T> result = *this;
	return result -= otherMat;
}
template <typename T>
inline Mat<T> Mat<T>::operator*(const Mat<T> &otherMat) const {
	Mat<T> result(rows, otherMat.cols);
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < otherMat.cols; ++j) {
			T newElement = 0;
			for (size_t k = 0; k < cols; ++k) {
				newElement += elements[i][k] * otherMat(k, j);
			}
			result(i, j) = newElement;
		}
	}
	return result;
}
template <typename T>
inline Vec<T> Mat<T>::operator*(const Vec<T> &vec) const {
	Vec<T> result(rows);
	for (size_t i = 0; i < rows; ++i) {
		T newElement = 0;
		for (size_t j = 0; j < cols; ++j) {
			newElement += elements[i][j] * vec(j);
		}
		result(i) = newElement;
	}
	return result;
}

template <typename T>
inline Mat<T> Mat<T>::operator+(T value) {
	Mat<T> result = *this;
	return result += value;
}
template <typename T>
inline Mat<T> Mat<T>::operator-(T value) {
	Mat<T> result = *this;
	return result -= value;
}
template <typename T>
inline Mat<T> Mat<T>::operator*(T value) {
	Mat<T> result = *this;
	return result *= value;
}
template <typename T>
inline Mat<T> Mat<T>::operator/(T value) {
	Mat<T> result = *this;
	return result /= value;
}

template <typename T>
inline std::ostream &operator<<(std::ostream &outputStream, const Mat<T> &mat) {
	outputStream << "[";
	for (size_t i = 0; i < mat.rows; ++i) {
		for (size_t j = 0; j < mat.cols; ++j) {
			outputStream << mat.elements[i][j];
			if (j < mat.cols - 1) {
				outputStream << "  ";
			}
		}
		if (i < mat.rows - 1) {
			outputStream << std::endl << " ";
		}
	}
	outputStream << "]";
	return outputStream;
}

template <typename T>
inline Mat<T> Mat<T>::transpose() const {
	Mat result(cols, rows);
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			result(j, i) = elements[i][j];
		}
	}
	return result;
}

template <typename T>
template <typename U>
inline Mat<T> Mat<T>::powByElements(U power) const {
	Mat<T> result = *this;
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			result(i, j) = std::pow(result(i, j), power);
		}
	}
	return result;
}

template <typename T>
inline Mat<T> Mat<T>::multiplyByElements(const Mat<T> &otherMat) const {
	Mat<T> result = *this;
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			result(i, j) *= otherMat(i, j);
		}
	}
	return result;
}
template <typename T>
inline Mat<T> Mat<T>::divideByElements(const Mat<T> &otherMat) const {
	Mat<T> result = *this;
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			result(i, j) /= otherMat(i, j);
		}
	}
	return result;
}

template <typename T>
inline Vec<size_t> Mat<T>::maxRowByCol() const {
	Vec<size_t> result(cols);
	for (size_t i = 0; i < cols; ++i) {
		T maxVal = 0;
		size_t maxRow = 0;
		for (size_t j = 0; j < rows; ++j) {
			if (j == 0 || elements[j][i] > maxVal) {
				maxVal = elements[j][i];
				maxRow = j;
			}
		}
		result(i) = maxRow;
	}
	return result;
}

template <typename T>
inline Vec<T> Mat<T>::getCol(size_t col) const {
	Vec<T> result(rows);
	for (size_t i = 0; i < rows; ++i) {
		result(i) = elements[i][col];
	}
	return result;
}

template <typename T>
inline Vec<T> Mat<T>::getRow(size_t row) const {
	Vec<T> result(cols);
	for (size_t i = 0; i < cols; ++i) {
		result(i) = elements[row][i];
	}
	return result;
}

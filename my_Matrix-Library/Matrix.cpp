#define _SCL_SECURE_NO_WARNINGS
#include"Matrix.h"

template<typename T>
std::ostream &operator<<(std::ostream &os, const Matrix<T> &mat) {
	if (mat.rows == 0 || mat.cols == 0) {
		os << "";
	}
	else {
		for (int i = 0; i < mat.rows; i++) {
			for (int j = 0; j < mat.cols; j++) {
				int index = i * mat.cols + j;
				os << mat.m_data[index] << " ";
			}
			os << "\n";
		}
	}
	return os;
}

template <typename T>
Matrix<T>::Matrix():m_data(nullptr),rows(0),cols(0) {
}

template<typename T>
Matrix<T>::~Matrix() {
	if (m_data != nullptr) {
		delete[] m_data;
	}
}

template <typename T>
Matrix<T>::Matrix(int row, int col):rows(row),cols(col),m_data(new T[row*col]) {
	memset(m_data, 0, sizeof(T)*rows*cols);
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T> &other) {
	
	assert(other.m_data != nullptr);

	this->rows = other.rows;
	this->cols = other.cols;

	this->m_data = new T[this->rows*this->cols];

	std::copy(other.m_data, other.m_data + other.rows*other.cols, this->m_data);
}

template<typename T>
Matrix<T>::Matrix(const std::vector<std::vector<T>> &_data) {
	assert(_data.size() > 0);
	assert(_data[0].size() > 0);

	this->rows = _data.size();
	this->cols = _data[0].size();

	m_data = new T[rows*cols];

	int start = 0;
	for (int i = 0; i < _data.size(); i++) {
		std::copy(_data[i].begin(), _data[i].end(), m_data + start);
		start += cols;
	}
}

template<typename T>
Matrix<T> &Matrix<T>::operator=(const Matrix<T> &other) {
	assert(other.rows != 0 && other.cols != 0);

	this->rows = other.rows;
	this->cols = other.cols;

	if (m_data != nullptr) {
		delete[] m_data;
	}
	m_data = new T[rows*cols];
	std::copy(other.m_data, other.m_data + rows * cols, m_data);
	return *this;
}

template<typename T>
inline T &Matrix<T>::at(int row_index, int col_index) {

	assert(row_index < rows);
	assert(col_index < cols);

	int index = row_index * cols+col_index;
	return m_data[index];
}

template<typename T>
inline T &Matrix<T>::at(int row_index, int col_index) const{

	assert(row_index < rows);
	assert(col_index < cols);

	int index = row_index * cols + col_index;
	return m_data[index];
}

template<typename T>
Matrix<T> Matrix<T>::add(const Matrix<T> &other){

	 assert(this->rows == other.rows);
	 assert(this->cols == other.cols);

	 int len = rows * cols;
	 for (int i = 0; i < len; i++) {
		 m_data[i] += other.m_data[i];
	 }
	 return *this;
}

template<typename T>
Matrix<T> Matrix<T>::sub(const Matrix<T> &other){
	assert(this->rows == other.rows);
	assert(this->cols == other.cols);

	int len = rows * cols;
	for (int i = 0; i < len; i++) {
		m_data[i] -= other.m_data[i];
	}
	return *this;
}

template<typename T>
Matrix<T> Matrix<T>::mul(const Matrix<T> &other){
	assert(cols == other.rows);
	T *res_data = new T[rows*other.cols];
	for (int r = 0; r < rows; ++r) {
		std::thread t1(&Matrix<T>::vector_mul_vector, this, r, other, res_data); ++r;
		std::thread t2(&Matrix<T>::vector_mul_vector, this, r, other, res_data); ++r;
		std::thread t3(&Matrix<T>::vector_mul_vector, this, r, other, res_data); ++r;
		std::thread t4(&Matrix<T>::vector_mul_vector, this, r, other, res_data);
		t1.join();
		t2.join();
		t3.join();
		t4.join();
	}
	if (m_data != nullptr) {
		delete[] m_data;
	}
	m_data = res_data;
	cols = other.cols;
	return *this;
}

template<typename T>
Matrix<T> Matrix<T>::t() {
	Matrix<T> this_temp(*this)
	assert(rows > 0 && cols > 0);
	assert(m_data != nullptr);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < i; j++) {
			std::swap(this_temp.m_data[i*cols + j], this_temp.m_data[j*cols + i]);
		}
	}
	return this_temp;
}

template<typename T>
Matrix<T> Matrix<T>::inv() {
	Matrix<T> this_inv(rows, cols);
	//采用LU分解的方法，
	assert(rows > 0 && cols > 0);
	assert(m_data != nullptr);
	assert(rows == cols);
	int N = rows;
	Matrix<T> L(rows, cols);
	Matrix<T> U(rows, cols);

	////计算L和U;
	for (int i = 0; i < N; i++) {
		U.at(0, i) = this->at(0, i);
		L.at(i, 0) = this->at(i, 0) / U.at(0, i);
	}
	for (int r = 0; r < N; r++) {

		for (int j = r; j < N; j++) {
			float s = 0.0;
			for (int k = 0; k < r; k++) {
				s += L.at(r, k)*U.at(k, j);
			}
			U.at(r, j) = this->at(r, j) - s;
		}
		
		for (int i = r; i < N; i++) {
			float s = 0.0;
			for (int k = 0; k < r; k++) {
				s += L.at(i, k)*U.at(k, r);
			}
			L.at(i, r) = (this->at(i, r) - s) / (U.at(r, r));
		}
	}

	//L的逆矩阵；
	Matrix<T> L_inv(rows, cols);
	for (int i = 0; i < N; i++) {
		L_inv.at(i, i) = 1;
		for (int k = i + 1; k < N; k++) {
			for (int j = i; j <=k - 1; j++) {
				L_inv.at(k, i) = L_inv.at(k, i) - L.at(k, j)*L_inv.at(j, i);
			}
		}
	}

	//计算U的逆矩阵；
	Matrix<T> U_inv(rows, cols);
	for (int i = 0; i < N; i++) {
		U_inv.at(i, i) = 1 / U.at(i, i);
		for (int k = i - 1; k >= 0; k--) {
			float s = 0;
			for (int j = k + 1; j <= i; j++) {
				s += U.at(k, j)*U_inv.at(j, i);
			}
			U_inv.at(k, i) = -s / U.at(k, k);
		}
	}

	//计算原矩阵的逆矩阵;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			for (int k = 0; k < rows; k++) {
				this_inv.at(i, j) += U_inv.at(i, k)*L_inv.at(k, j);
			}
		}
	}

	
	U.clear();
	U_inv.clear();
	L.clear();
	L_inv.clear();
	return this_inv;
}

template<typename T>
Matrix<T> Matrix<T>::zero() {
	if (rows != 0 && cols != 0&&m_data!=nullptr) {
		memset(this->m_data,0, sizeof(T)*rows*cols)
	}
	return *this;
}

template<typename T>
Matrix<T> Matrix<T>::zero(int row,int col) {

	if (m_data != nullptr) {
		delete[] m_data;
	}

	rows = row;
	cols = col;

	m_data = new T[rows*cols];
	memset(this->m_data, 0, sizeof(T)*rows*cols);
	return *this;
}
template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &other) {
	assert(this->rows == other.rows);
	assert(this->cols == other.cols);

	Matrix<T> temp(*this);
	int len = rows * cols;
	for (int i = 0; i < len; i++) {
		temp.m_data[i]+=other.m_data[i];
	}
	return temp;
}
template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T> &other) {
	assert(this->rows == other.rows);
	assert(this->cols == other.cols);

	Matrix<T> temp(*this);
	int len = rows * cols;
	for (int i = 0; i < len; i++) {
		temp.m_data[i] -= other.m_data[i];
	}
	return temp;
}
template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &other) {
	assert(this->cols == other.rows);

	Matrix<T> temp(rows,other.cols);
	for (int r = 0; r < rows; ++r) {
		std::thread t1(&Matrix<T>::vector_mul_vector, this, r, other, temp.m_data); ++r;
		std::thread t2(&Matrix<T>::vector_mul_vector, this, r, other, temp.m_data); ++r;
		std::thread t3(&Matrix<T>::vector_mul_vector, this, r, other, temp.m_data); ++r;
		std::thread t4(&Matrix<T>::vector_mul_vector, this, r, other, temp.m_data);
		t1.join();
		t2.join();
		t3.join();
		t4.join();
	}
	return temp;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const T &const_temp) {
	Matrix<T> mat(rows,cols);

	int len = mat.rows*mat.cols;

	for (int i = 0; i < len; i++) {
		mat.m_data[i] = (T)this->m_data[i] * const_temp;
	}
	return mat;
}

template<typename T>
Matrix<T> Matrix<T>::operator/(const T &const_temp) {
	Matrix<T> mat(rows, cols);

	int len = mat.rows*mat.cols;

	for (int i = 0; i < len; i++) {
		mat.m_data[i] = (T)this->m_data[i] / const_temp;
	}
	return mat;
}

template<typename T>
void Matrix<T>::clear() {
	if (m_data != nullptr) {
		delete[] m_data;
		m_data = nullptr;
		rows = 0;
		cols = 0;
	}
}

template<typename T>
void Matrix<T>::vector_mul_vector(int row, const Matrix<T> &mat, T *data) {
	assert(data != nullptr);
	if (row >= rows) {
		return;
	}
	int data_index = row * mat.cols;
	for (int mat_col = 0; mat_col < mat.cols; ++mat_col) {
		int start_index = row*cols;
		T sum_temp = 0;
		for (int mat_row = 0; mat_row < mat.rows; ++mat_row) {
			sum_temp += m_data[start_index] * mat.at(mat_row, mat_col);
			++start_index;
		}
		data[data_index] = sum_temp;
		++data_index;
	}
}

template<typename T>
void Matrix<T>::push_back(std::vector<T> vec) {
	if (m_data != nullptr) {
		assert(vec.size() == cols);
		int len = (rows + 1) * (cols);
		T *temp_data = new T[len];
		std::copy(m_data, m_data + rows * cols, temp_data);
		std::copy(vec.begin(), vec.end(), temp_data + rows * cols);
		delete[] m_data;
		rows += 1;
		m_data = temp_data;
	}
	else {
		rows = 1;
		cols = vec.size();

		m_data = new T[cols];
		std::copy(vec.begin(), vec.end(), m_data );
	}
}

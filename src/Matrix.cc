#include <cnn/Matrix.h>
#include <cnn/Common.h>
#include <stdexcept>

namespace cnn{

Matrix::Matrix(size_t rows, size_t cols)
    : rows_(rows),
      cols_(cols), 
      data_(rows * cols, 0.0){}

Matrix::Matrix(size_t rows, size_t cols, double value)
    : rows_(rows),
      cols_(cols), 
      data_((rows * cols), value){}

void Matrix::checkInitializerList(const std::initializer_list<std::initializer_list<double>>& values) const{
  for(const auto& row : values){
    if(row.size() != cols_){
      throw std::invalid_argument("All rows in initializer_list must have the same number of columns.");
    }
  }

  if(cols_ == 0){
    throw std::invalid_argument("Initializer list must not contain empty rows.");
  }
}

Matrix::Matrix(std::initializer_list<std::initializer_list<double>> values)
    : rows_(values.size()),
      cols_(values.size() == 0 ? 0 : values.begin()->size()),
      data_(rows_ * cols_){
  if(rows_ == 0){
    return;
  }
  checkInitializerList(values);

  size_t index{0};
  for(const auto& row : values){
    for(const double value : row){
      data_[index++] = value;
    }
  }
}

size_t Matrix::rows() const noexcept{
  return rows_;
}

size_t Matrix::cols() const noexcept{
  return cols_;
}

size_t Matrix::index(size_t row, size_t col) const noexcept{
  return row * cols_ + col;
}

size_t Matrix::size() const noexcept{
  return rows_ * cols_;
}

bool Matrix::empty() const noexcept{
  return size() == 0;
}

double& Matrix::operator()(size_t row, size_t col){
  if(row >= rows_ || col >= cols_){
    throw std::out_of_range("Matrix index out of range");
  }
  return data_[index(row, col)];
}

const double& Matrix::operator()(size_t row, size_t col) const{
  if(row >= rows_ || col >= cols_){
    throw std::out_of_range("Matrix index out of range");
  }
  return data_[index(row, col)];
}

void Matrix::checkSameDimensions(const Matrix& other) const{
  if(rows_ != other.rows_ || cols_ != other.cols_){
    throw std::invalid_argument("Matrices must have the same dimensions.");
  }
}

void Matrix::checkMultiplicationDimensions(const Matrix& other) const{
  if(cols_ != other.rows_){
    throw std::invalid_argument("Matrices must have appropriate dimensions.");
  }
}

Matrix& Matrix::operator+=(const Matrix& other){
  checkSameDimensions(other);

  for(size_t i{0}; i<size(); ++i){
    data_[i] += other.data_[i];
  }

  return *this;
}

Matrix Matrix::operator+(const Matrix& other) const{
  Matrix result(*this);
  result += other;
  return result;
}

Matrix& Matrix::operator-=(const Matrix& other){
  checkSameDimensions(other);

  for(size_t i{0}; i<size(); ++i){
    data_[i] -= other.data_[i];
  }

  return *this;
}

Matrix Matrix::operator-(const Matrix& other) const{
  Matrix result(*this);
  result -= other;
  return result;
}

Matrix& Matrix::operator*=(double scalar){
  for(size_t i{0}; i<size(); ++i){
    data_[i] *= scalar;
  }

  return *this;
}

Matrix Matrix::operator*(double scalar) const{
  Matrix result(*this);
  result *= scalar;
  return result;
}

Matrix Matrix::operator*(const Matrix& other) const{
  checkMultiplicationDimensions(other);
  Matrix result(rows_, other.cols_);

  for(size_t r{0}; r<rows_; ++r){
    for(size_t c{0}; c<other.cols_; ++c){
      double sum{0.0};
      for(size_t i{0}; i<cols_; ++i){
        sum+=(operator()(r,i)*other(i,c));
      }
      result(r,c) = sum;
    }
  }

  return result;
}

bool Matrix::operator==(const Matrix& other) const noexcept{
  if(rows_ != other.rows_ || cols_ != other.cols_){
    return false;
  }

  for(size_t i{0}; i<size(); ++i){
    printf("%f\n", data_[i]);
  }
      printf("\n");


  for(size_t i{0}; i<size(); ++i){
    if(std::abs(data_[i] - other.data_[i]) > kEpsilon){
      return false;
    }
  }

  return true;
}

bool Matrix::operator!=(const Matrix& other) const noexcept{
  return !(*this == other);
}

Matrix Matrix::transpose() const{
  Matrix result(cols_, rows_);

  for(size_t r{0}; r<rows_; ++r){
    for(size_t c{0}; c<cols_; ++c){
      result(c,r) = (*this)(r,c);
    }
  }

  return result;
}

Matrix Matrix::hadamard(const Matrix& other) const{
  checkSameDimensions(other);

  Matrix result(rows_, cols_);

  for(size_t i{0}; i<size(); ++i){
    result.data_[i] = data_[i] * other.data_[i];
  }

  return result;
}

} // namespace cnn
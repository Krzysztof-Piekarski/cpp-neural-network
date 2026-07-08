#include <cnn/Matrix.h>
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

Matrix Matrix::operator+(const Matrix& other) const{
  checkSameDimensions(other);
  Matrix result(rows_, cols_);

  for(size_t i{0}; i<size(); ++i){
    result.data_[i] = data_[i] + other.data_[i];
  }

  return result;
}

Matrix Matrix::operator-(const Matrix& other) const{
  checkSameDimensions(other);
  Matrix result(rows_, cols_);

  for(size_t i{0}; i<size(); ++i){
    result.data_[i] = data_[i] - other.data_[i];
  }

  return result;
}

void Matrix::checkSameDimensions(const Matrix& other) const{
  if(rows_ != other.rows_ || cols_ != other.cols_){
    throw std::invalid_argument("Matrices must have the same dimensions.");
  }
}

void Matrix::checkMultiplicationDimensions(const Matrix& other) const{
  if(cols_ != other.rows_){
    throw std::invalid_argument("Matrices must have the same dimensions.");
  }
}

} // namespace cnn
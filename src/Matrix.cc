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

} // namespace cnn
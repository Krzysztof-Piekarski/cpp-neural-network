#include <cnn/Matrix.h>

namespace cnn{

  Matrix::Matrix(size_t rows, size_t cols)
      : rows_(rows),
        cols_(cols), 
        data_(rows * cols){
  }

  size_t Matrix::rows(){
    return rows_;
  }

  size_t Matrix::cols(){
    return cols_;
  }

}
#pragma once

#include <cnn/Matrix.h>

namespace cnn::initializer{

Matrix zeros(size_t rows, size_t cols);

Matrix ones(size_t rows, size_t cols);

Matrix random(size_t rows, size_t cols, double min = -1.0, double max = 1.0);

} // namespace cnn::initializer
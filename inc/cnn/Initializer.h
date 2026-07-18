#pragma once

#include <cnn/Matrix.h>

namespace cnn::initializer{

Matrix zeros(size_t, size_t);

Matrix ones(size_t, size_t);

Matrix random(size_t, size_t, double min = -1.0, double max = 1.0);

} // namespace cnn::initializer
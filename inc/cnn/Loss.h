#pragma once

#include <cnn/Matrix.h>

namespace cnn::loss{

double meanSquaredError(const Matrix& prediction,
                        const Matrix& target);

Matrix meanSquaredErrorDerivative(const Matrix& prediction,
                                  const Matrix& target);

} // namespace cnn::loss
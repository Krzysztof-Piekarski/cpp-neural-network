#pragma once

#include <cnn/Types.h>

namespace cnn::activation{

double sigmoid(double x) noexcept;
double sigmoidDerivative(double x) noexcept;

double relu(double x) noexcept;
double reluDerivative(double x) noexcept;

double tanh(double x) noexcept;
double tanhDerivative(double x) noexcept;

double apply(double x, ActivationType type);
double derivative(double x, ActivationType type);

} // namespace cnn::activation
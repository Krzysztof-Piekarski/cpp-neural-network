#pragma once

namespace cnn::activation{

double sigmoid(double x) noexcept;

double sigmoidDerivative(double x) noexcept;

double relu(double x) noexcept;

double reluDerivative(double x) noexcept;

double tanh(double x) noexcept;

double tanhDerivative(double x) noexcept;

} // namespace cnn::activation
#pragma once

namespace cnn::activation{

double sigmoid(double) noexcept;

double sigmoidDerivative(double) noexcept;

double relu(double) noexcept;

double reluDerivative(double) noexcept;

double tanh(double) noexcept;

double tanhDerivative(double) noexcept;

} // namespace cnn::activation
#pragma once

namespace cnn::activation{

double sigmoid(double) noexcept;

double relu(double) noexcept;

double tanh(double);

double sigmoidDerivative(double);

double reluDerivative(double);

} // namespace cnn::activation
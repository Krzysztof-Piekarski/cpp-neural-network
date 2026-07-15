#pragma once

namespace cnn::activation{

double sigmoid(double) noexcept;

double relu(double);

double tanh(double);

double sigmoidDerivative(double);

double reluDerivative(double);

} // namespace cnn::activation
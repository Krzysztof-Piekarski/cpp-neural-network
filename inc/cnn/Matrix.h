#pragma once

#include <cstddef>
#include <vector>

namespace cnn{

class Matrix{
public:
    Matrix() = default;
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, double value);
    Matrix(std::initializer_list<std::initializer_list<double>> values);

    size_t rows() const noexcept;
    size_t cols() const noexcept;
    size_t size() const noexcept;
    bool empty() const noexcept;

    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;

    Matrix& operator+=(const Matrix& other);
    Matrix operator+(const Matrix& other) const;
    Matrix& operator-=(const Matrix& other);
    Matrix operator-(const Matrix& other) const;
    Matrix& operator*=(double scalar);
    Matrix operator*(double scalar) const;
    Matrix operator*(const Matrix& other) const;

    bool operator==(const Matrix& other) const noexcept;
    bool operator!=(const Matrix& other) const noexcept;

    Matrix transpose() const;
    template<typename Function>
    Matrix map(Function fn) const;

private:
    void checkInitializerList(const std::initializer_list<std::initializer_list<double>>& values) const;
    void checkSameDimensions(const Matrix& other) const;
    void checkMultiplicationDimensions(const Matrix& other) const;

    size_t rows_{0};
    size_t cols_{0};
    size_t index(size_t row, size_t col) const noexcept;
    std::vector<double> data_;
};

} // namespace cnn

#include <cnn/Matrix.tpp>
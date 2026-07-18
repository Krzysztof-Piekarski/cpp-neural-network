#include <cnn/Initializer.h>
#include <random>

namespace cnn::initializer{

Matrix zeros(size_t rows, size_t cols){
    return Matrix(rows, cols, 0.0);
}

Matrix ones(size_t rows, size_t cols){
    return Matrix(rows, cols, 1.0);
}

Matrix random(size_t rows,
              size_t cols,
              double min,
              double max){
    if(min > max){
        throw std::invalid_argument(
            "Minimum value must not exceed maximum value.");
    }

    Matrix result(rows, cols);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min, max);

    for(size_t r{0}; r<result.rows(); ++r){
        for(size_t c{0}; c<result.cols(); ++c){
            result(r,c) = dist(gen);
        }
    }

    return result;
}

} // namespace cnn::initializer
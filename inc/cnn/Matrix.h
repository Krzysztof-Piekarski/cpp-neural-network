#include <vector>

namespace cnn{

    class Matrix{
    public:
    Matrix(size_t rows, size_t cols);
    size_t rows();
    size_t cols();
    
    private:
    size_t rows_;
    size_t cols_;
    std::vector<double> data_;
    };

}
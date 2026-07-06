#include <cstddef>
#include <vector>

namespace cnn{

class Matrix{
public:
Matrix() = default;
Matrix(size_t rows, size_t cols);
Matrix(size_t rows, size_t cols, double value);

size_t rows() const noexcept;
size_t cols() const noexcept;
size_t size() const noexcept;
bool empty() const noexcept;

double& operator()(size_t row, size_t col);
const double& operator()(size_t row, size_t col) const;

private:
size_t rows_{0};
size_t cols_{0};
size_t index(size_t row, size_t col) const noexcept;
size_t size_{0};
std::vector<double> data_;
};

} // namespace cnn
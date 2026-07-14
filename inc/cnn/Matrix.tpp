#pragma once

namespace cnn{

template<typename Function>
Matrix Matrix::map(Function fn) const{
    Matrix result(rows_, cols_);

    for(size_t i{0}; i < size(); ++i){
        result.data_[i] = fn(data_[i]);
    }

    return result;
}

} // namespace cnn
#include <gtest/gtest.h>
#include <cnn/Matrix.h>

TEST(Matrix, ConstructorStoresSize)
{
    cnn::Matrix matrix(3,4);

    EXPECT_EQ(matrix.rows(), 3);
    EXPECT_EQ(matrix.cols(), 4);
}
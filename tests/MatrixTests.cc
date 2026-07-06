#include <gtest/gtest.h>
#include <cnn/Matrix.h>

TEST(Matrix, ConstructorStoresDimensions){
    cnn::Matrix matrix(2,4);

    EXPECT_EQ(matrix.rows(), 2);
    EXPECT_EQ(matrix.cols(), 4);
}

TEST(Matrix, SizeReturnsNumberOfElements){
    cnn::Matrix matrix(2,4);

    EXPECT_EQ(matrix.size(), 8);
    EXPECT_FALSE(matrix.empty());
}

TEST(Matrix, EmptyReturnsTrueForEmptyMatrix){
    cnn::Matrix matrix(0,0);

    EXPECT_TRUE(matrix.empty());
}

TEST(Matrix, ConstSizeAndEmpty){
    const cnn::Matrix matrix(4,4);

    EXPECT_EQ(matrix.size(), 16);
    EXPECT_FALSE(matrix.empty());
}

TEST(Matrix, InitializesWithZero){
    cnn::Matrix matrix(2,2);

    EXPECT_DOUBLE_EQ(matrix(1,0), 0.0);
}

TEST(Matrix, CanWriteAndReadElement){
    cnn::Matrix matrix(2,4);
    EXPECT_DOUBLE_EQ(matrix(1,2), 0.0);

    matrix(1,2) = 4.4;
    
    EXPECT_DOUBLE_EQ(matrix(1,2), 4.4);
}

TEST(Matrix, ConstAccess){
    const cnn::Matrix matrix(2,2,7.0);

    EXPECT_DOUBLE_EQ(matrix(1,1), 7.0);
}

TEST(Matrix, AccessOutsideBoundsThrows){
    cnn::Matrix matrix(1,1);

    EXPECT_THROW(matrix(1,1), std::out_of_range);

    const cnn::Matrix constMatrix(2,3);

    EXPECT_THROW(constMatrix(1,3), std::out_of_range);
}

TEST(Matrix, InitializesAllElementsWithValue){
    cnn::Matrix matrix(2, 3, 8.88);

    EXPECT_DOUBLE_EQ(matrix(0,0), 8.88);
    EXPECT_DOUBLE_EQ(matrix(0,2), 8.88);
    EXPECT_DOUBLE_EQ(matrix(1,2), 8.88);
}
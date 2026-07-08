#include <gtest/gtest.h>
#include <cnn/Matrix.h>

void FillMatrix(cnn::Matrix& m){
    double value = 1.0;

    for(size_t r{0}; r<m.rows(); ++r){
        for(size_t c{0}; c<m.cols(); ++c){
            m(r,c) = value++;
        }
    }
}

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

TEST(Matrix, CopyConstructorCreatesIndependentCopy){
    cnn::Matrix matrix1(2,2);
    FillMatrix(matrix1);
    cnn::Matrix matrix2 = matrix1;

    EXPECT_EQ(matrix1.rows(), matrix2.rows());
    EXPECT_EQ(matrix1.cols(), matrix2.cols());
    EXPECT_EQ(matrix2.size(), 4);
    EXPECT_DOUBLE_EQ(matrix2(0,0), 1.0);
    EXPECT_DOUBLE_EQ(matrix2(0,1), 2.0);
    EXPECT_DOUBLE_EQ(matrix2(1,0), 3.0);
    EXPECT_DOUBLE_EQ(matrix2(1,1), 4.0);

    matrix1(1,1) = 0.0;

    EXPECT_DOUBLE_EQ(matrix1(1,1), 0.0);
    EXPECT_DOUBLE_EQ(matrix2(1,1), 4.0);

    matrix2(0,0) = 2.2;

    EXPECT_DOUBLE_EQ(matrix1(0,0), 1.0);
    EXPECT_DOUBLE_EQ(matrix2(0,0), 2.2);
}

TEST(Matrix, CopyAssignmentCreatesIndependentCopy){
    cnn::Matrix matrix1(2,2, 1.1);
    cnn::Matrix matrix2(3,3, 3.3);

    EXPECT_EQ(matrix2.size(), 9);
    EXPECT_DOUBLE_EQ(matrix1(0,0), 1.1);
    EXPECT_DOUBLE_EQ(matrix2(0,0), 3.3);

    matrix2 = matrix1;

    EXPECT_EQ(matrix1.rows(), matrix2.rows());
    EXPECT_EQ(matrix1.cols(), matrix2.cols());
    EXPECT_EQ(matrix2.size(), 4);
    EXPECT_DOUBLE_EQ(matrix1(0,0), 1.1);
    EXPECT_DOUBLE_EQ(matrix2(0,0), 1.1);

    matrix1(1,1) = 0.0;

    EXPECT_DOUBLE_EQ(matrix1(1,1), 0.0);
    EXPECT_DOUBLE_EQ(matrix2(1,1), 1.1);

    matrix2(0,0) = 2.2;

    EXPECT_DOUBLE_EQ(matrix1(0,0), 1.1);
    EXPECT_DOUBLE_EQ(matrix2(0,0), 2.2);
}

TEST(Matrix, Addition){
    cnn::Matrix a(2,2, 1.1);
    cnn::Matrix b(2,2, 3.3);

    cnn::Matrix c = a + b;

    EXPECT_DOUBLE_EQ(c(0,0),4.4);
    EXPECT_DOUBLE_EQ(c(0,1),4.4);
    EXPECT_DOUBLE_EQ(c(1,0),4.4);
    EXPECT_DOUBLE_EQ(c(1,1),4.4);
}

TEST(Matrix, AdditionThrowsForDifferentDimensions){
    cnn::Matrix a(2,3);
    cnn::Matrix b(3,2);

    EXPECT_THROW(a + b, std::invalid_argument);
}

TEST(Matrix, Subtraction){
    cnn::Matrix a(2,2, 3.3);
    cnn::Matrix b(2,2, 1.1);

    cnn::Matrix c = a - b;

    EXPECT_DOUBLE_EQ(c(0,0),2.2);
    EXPECT_DOUBLE_EQ(c(0,1),2.2);
    EXPECT_DOUBLE_EQ(c(1,0),2.2);
    EXPECT_DOUBLE_EQ(c(1,1),2.2);
}

TEST(Matrix, SubtractionThrowsForDifferentDimensions){
    cnn::Matrix a(2,3);
    cnn::Matrix b(3,2);

    EXPECT_THROW(a - b, std::invalid_argument);
}
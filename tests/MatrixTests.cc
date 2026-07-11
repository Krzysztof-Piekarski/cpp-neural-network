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

TEST(Matrix, AdditionThrowsForDifferentDimensions){
    cnn::Matrix a(2,3);
    cnn::Matrix b(3,2);

    EXPECT_THROW(a + b, std::invalid_argument);
    EXPECT_THROW(a += b, std::invalid_argument);
}

TEST(Matrix, AdditionAssignment){
    cnn::Matrix a(2,2, 1.1);
    cnn::Matrix b(2,2, 3.3);

    a += b;

    EXPECT_DOUBLE_EQ(a(0,0),4.4);
    EXPECT_DOUBLE_EQ(a(0,1),4.4);
    EXPECT_DOUBLE_EQ(a(1,0),4.4);
    EXPECT_DOUBLE_EQ(a(1,1),4.4);

    EXPECT_DOUBLE_EQ(b(0,0),3.3);
}

TEST(Matrix, Addition){
    cnn::Matrix a(2,2, 1.1);
    cnn::Matrix b(2,2, 3.3);

    cnn::Matrix c = a + b;

    EXPECT_DOUBLE_EQ(c(0,0),4.4);
    EXPECT_DOUBLE_EQ(c(0,1),4.4);
    EXPECT_DOUBLE_EQ(c(1,0),4.4);
    EXPECT_DOUBLE_EQ(c(1,1),4.4);

    EXPECT_DOUBLE_EQ(a(0,0),1.1);
    EXPECT_DOUBLE_EQ(b(0,0),3.3);
}

TEST(Matrix, AdditionAssignmentReturnsReference){
    cnn::Matrix a(2,2, 1.1);
    cnn::Matrix b(2,2, 2.2);
    cnn::Matrix c(2,2, 3.3);

    (a += b) += c;

    EXPECT_DOUBLE_EQ(a(0,0),6.6);
}

TEST(Matrix, SubtractionThrowsForDifferentDimensions){
    cnn::Matrix a(2,3);
    cnn::Matrix b(3,2);

    EXPECT_THROW(a - b, std::invalid_argument);
    EXPECT_THROW(a -= b, std::invalid_argument);
}

TEST(Matrix, SubtractionAssignment){
    cnn::Matrix a(2,2, 3.3);
    cnn::Matrix b(2,2, 1.1);

    a -= b;

    EXPECT_DOUBLE_EQ(a(0,0),2.2);
    EXPECT_DOUBLE_EQ(a(0,1),2.2);
    EXPECT_DOUBLE_EQ(a(1,0),2.2);
    EXPECT_DOUBLE_EQ(a(1,1),2.2);

    EXPECT_DOUBLE_EQ(b(1,1),1.1);
}

TEST(Matrix, Subtraction){
    cnn::Matrix a(2,2, 3.3);
    cnn::Matrix b(2,2, 1.1);

    cnn::Matrix c = a - b;

    EXPECT_DOUBLE_EQ(c(0,0),2.2);
    EXPECT_DOUBLE_EQ(c(0,1),2.2);
    EXPECT_DOUBLE_EQ(c(1,0),2.2);
    EXPECT_DOUBLE_EQ(c(1,1),2.2);

    EXPECT_DOUBLE_EQ(a(0,0),3.3);
    EXPECT_DOUBLE_EQ(b(0,0),1.1);
}

TEST(Matrix, SubtractionAssignmentReturnsReference){
    cnn::Matrix a(2,2, 1.0);
    cnn::Matrix b(2,2, 2.0);
    cnn::Matrix c(2,2, 3.0);

    (a -= b) -= c;

    EXPECT_DOUBLE_EQ(a(0,0),-4.0);
}

TEST(Matrix, ScalarMultiplicationAssignment){
    cnn::Matrix matrix(2,2);
    FillMatrix(matrix);

    matrix *= 2.0;

    EXPECT_DOUBLE_EQ(matrix(0,0), 2.0);
    EXPECT_DOUBLE_EQ(matrix(0,1), 4.0);
    EXPECT_DOUBLE_EQ(matrix(1,0), 6.0);
    EXPECT_DOUBLE_EQ(matrix(1,1), 8.0);

    matrix *= 2u;

    EXPECT_DOUBLE_EQ(matrix(0,0), 4.0);

    matrix *= 0.25f;

    EXPECT_DOUBLE_EQ(matrix(0,0), 1.0);
}

TEST(Matrix, ScalarMultiplication){
    cnn::Matrix matrix(2,2);
    FillMatrix(matrix);

    cnn::Matrix matrix2 = matrix * 2.0;

    EXPECT_DOUBLE_EQ(matrix2(0,0), 2.0);
    EXPECT_DOUBLE_EQ(matrix2(0,1), 4.0);
    EXPECT_DOUBLE_EQ(matrix2(1,0), 6.0);
    EXPECT_DOUBLE_EQ(matrix2(1,1), 8.0);

    matrix = matrix * 2u;

    EXPECT_DOUBLE_EQ(matrix(0,0), 2.0);

    matrix = matrix2 * 0.25f;

    EXPECT_DOUBLE_EQ(matrix(0,0), 0.5);
}

TEST(Matrix, ScalarMultiplicationAssignmentReturnsReference){
    cnn::Matrix a(2,2, 1.0);

    (a *= 2.0) *= 4.f;

    EXPECT_DOUBLE_EQ(a(0,0),8.0);
}

TEST(Matrix, MultiplicationThrowsForIncorrectDimensions){
    cnn::Matrix a(3,2);
    cnn::Matrix b(3,2);

    EXPECT_THROW(a * b, std::invalid_argument);
}

TEST(Matrix, Multiplication){
    cnn::Matrix a(3,2);
    cnn::Matrix b(2,3);

    a(0,0) = 8.1;
    a(0,1) = 10.8;
    a(1,0) = -22.9;
    a(1,1) = -13.7;
    a(2,0) = 24.2;
    a(2,1) = 16.4;

    b(0,0) = 33.8;
    b(0,1) = -42.0;
    b(0,2) = 777.0;
    b(1,0) = 19.10;
    b(1,1) = 15.5;
    b(1,2) = 2.1;
    

    cnn::Matrix c = a * b;

    ASSERT_EQ(c.rows(),3);
    ASSERT_EQ(c.cols(),3);

    constexpr double eps = 1e-9;
    EXPECT_NEAR(c(0,0),480.06,eps);
    EXPECT_NEAR(c(0,1),-172.8,eps);
    EXPECT_NEAR(c(0,2),6316.38,eps);
    EXPECT_NEAR(c(1,0),-1035.69,eps);
    EXPECT_NEAR(c(1,1),749.45,eps);
    EXPECT_NEAR(c(1,2),-17822.07,eps);
    EXPECT_NEAR(c(2,0),1131.2,eps);
    EXPECT_NEAR(c(2,1),-762.2,eps);
    EXPECT_NEAR(c(2,2),18837.84,eps);

    EXPECT_DOUBLE_EQ(a(0,0),8.1);
    EXPECT_DOUBLE_EQ(b(0,0),33.8);
    EXPECT_DOUBLE_EQ(a(1,1),-13.7);
    EXPECT_DOUBLE_EQ(b(1,1),15.5);
    EXPECT_DOUBLE_EQ(a(2,1),16.4);
    EXPECT_DOUBLE_EQ(b(1,2),2.1);
    
}

TEST(Matrix, Multiplication1x1){
    cnn::Matrix a(1,1, 2.0);
    cnn::Matrix b(1,1, 3.0);

    cnn::Matrix c = a * b;

    ASSERT_EQ(c.rows(),1);
    ASSERT_EQ(c.cols(),1);

    EXPECT_DOUBLE_EQ(c(0,0),6.0);
}

TEST(Matrix, MultiplicationByZero){
    cnn::Matrix a(2,2, 2.0);
    cnn::Matrix b(2,2, 0.0);

    cnn::Matrix c = a * b;

    EXPECT_DOUBLE_EQ(c(0,0), 0.0);
    EXPECT_DOUBLE_EQ(c(0,1), 0.0);
    EXPECT_DOUBLE_EQ(c(1,0), 0.0);
    EXPECT_DOUBLE_EQ(c(1,1), 0.0);
}

TEST(Matrix, MultiplicationOfTheSameMatrices){
    cnn::Matrix a(3,3);
    FillMatrix(a);
    cnn::Matrix b = a;

    cnn::Matrix c = a * b;

    EXPECT_DOUBLE_EQ(c(0,0), 30.0);
    EXPECT_DOUBLE_EQ(c(0,1), 36.0);
    EXPECT_DOUBLE_EQ(c(0,2), 42.0);
    EXPECT_DOUBLE_EQ(c(1,0), 66.0);
    EXPECT_DOUBLE_EQ(c(1,1), 81.0);
    EXPECT_DOUBLE_EQ(c(1,2), 96.0);
    EXPECT_DOUBLE_EQ(c(2,0), 102.0);
    EXPECT_DOUBLE_EQ(c(2,1), 126.0);
    EXPECT_DOUBLE_EQ(c(2,2), 150.0);

    EXPECT_DOUBLE_EQ(a(1,1), 5.0);
    EXPECT_DOUBLE_EQ(b(1,1), 5.0);
}

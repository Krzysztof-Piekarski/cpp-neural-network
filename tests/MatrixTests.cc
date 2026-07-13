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

TEST(Matrix, ConstructorWithInitializerListThrowsForIncorrectDimensions){
    EXPECT_THROW(cnn::Matrix matrix({{3.0, 5.0},
                                     {4.0, 8.0},
                                     {9.0, 4.2, 9.9}}), std::invalid_argument);
}

TEST(Matrix, ConstructorWithInitializerListThrowsForEmptyColumn){
    EXPECT_THROW(cnn::Matrix matrix{{}}, std::invalid_argument);
}

TEST(Matrix, ConstructorWithInitializerListStoresValues){
    cnn::Matrix matrix{{3.0, 5.0},
                       {4.0, 8.0},
                       {9.0, 4.2}};

    EXPECT_EQ(matrix.size(), 6);
    EXPECT_EQ(matrix.rows(), 3);
    EXPECT_EQ(matrix.cols(), 2);

    EXPECT_DOUBLE_EQ(matrix(0,0), 3.0);
    EXPECT_DOUBLE_EQ(matrix(0,1), 5.0);
    EXPECT_DOUBLE_EQ(matrix(1,0), 4.0);
    EXPECT_DOUBLE_EQ(matrix(1,1), 8.0);
    EXPECT_DOUBLE_EQ(matrix(2,0), 9.0);
    EXPECT_DOUBLE_EQ(matrix(2,1), 4.2);
}

TEST(Matrix, EmptyMatrixFromConstructorWithInitializerList){
    cnn::Matrix matrix{};

    EXPECT_TRUE(matrix.empty());
    EXPECT_EQ(matrix.rows(), 0);
    EXPECT_EQ(matrix.cols(), 0);
}

TEST(Matrix, MatrixWithOneValue){
    cnn::Matrix matrix{{1.0}};

    EXPECT_FALSE(matrix.empty());
    EXPECT_EQ(matrix.rows(), 1);
    EXPECT_EQ(matrix.cols(), 1);
}

TEST(Matrix, EqualityOperator){
    cnn::Matrix matrix{{1.0, 2.0},
                       {3.0, 5.0}};

    cnn::Matrix expected{{1.0, 2.0},
                         {3.0, 5.0}};

    EXPECT_TRUE(matrix == expected);
    EXPECT_FALSE(matrix != expected);

    expected(1,1) = 4.0;

    EXPECT_FALSE(matrix == expected);
    EXPECT_TRUE(matrix != expected);
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
    cnn::Matrix expected(2,2, 4.4);

    a += b;

    EXPECT_TRUE(a == expected);
    EXPECT_DOUBLE_EQ(b(0,0), 3.3);
}

TEST(Matrix, Addition){
    cnn::Matrix a(2,2, 1.1);
    cnn::Matrix b(2,2, 3.3);
    cnn::Matrix expected(2,2, 4.4);

    cnn::Matrix c = a + b;

    EXPECT_TRUE(c == expected);

    EXPECT_DOUBLE_EQ(a(0,0), 1.1);
    EXPECT_DOUBLE_EQ(b(0,0), 3.3);
}

TEST(Matrix, AdditionAssignmentReturnsReference){
    cnn::Matrix a(2,2, 1.1);
    cnn::Matrix b(2,2, 2.2);
    cnn::Matrix c(2,2, 3.3);

    (a += b) += c;

    EXPECT_DOUBLE_EQ(a(0,0), 6.6);
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
    cnn::Matrix expected(2,2, 2.2);

    a -= b;

    EXPECT_TRUE(a == expected);

    EXPECT_DOUBLE_EQ(b(1,1),1.1);
}

TEST(Matrix, Subtraction){
    cnn::Matrix a(2,2, 3.3);
    cnn::Matrix b(2,2, 1.1);
    cnn::Matrix expected(2,2, 2.2);

    cnn::Matrix c = a - b;

    EXPECT_TRUE(c == expected);

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
    cnn::Matrix expected{{2.0, 4.0},
                          {6.0, 8.0}};

    matrix *= 2.0;

    EXPECT_TRUE(matrix == expected);

    matrix *= 2u;

    EXPECT_DOUBLE_EQ(matrix(0,0), 4.0);

    matrix *= 0.25f;

    EXPECT_DOUBLE_EQ(matrix(0,0), 1.0);
}

TEST(Matrix, ScalarMultiplication){
    cnn::Matrix matrix(2,2);
    FillMatrix(matrix);
    cnn::Matrix expected{{2.0, 4.0},
                          {6.0, 8.0}};

    cnn::Matrix matrix2 = matrix * 2.0;

    EXPECT_TRUE(matrix2 == expected);

    matrix = matrix * 2u;

    EXPECT_DOUBLE_EQ(matrix(0,0), 2.0);

    matrix = matrix2 * 0.25f;

    EXPECT_DOUBLE_EQ(matrix(0,0), 0.5);
}

TEST(Matrix, ScalarMultiplicationAssignmentReturnsReference){
    cnn::Matrix a(2,2, 1.0);

    (a *= 2.0) *= 4.f;

    EXPECT_DOUBLE_EQ(a(0,0), 8.0);
}

TEST(Matrix, MultiplicationThrowsForIncorrectDimensions){
    cnn::Matrix a(3,2);
    cnn::Matrix b(3,2);

    EXPECT_THROW(a * b, std::invalid_argument);
}

TEST(Matrix, Multiplication){
    cnn::Matrix a{{8.1, 10.8},
                  {-22.4, -13.8},
                  {24.2, 16.4}};
    cnn::Matrix b{{33.8, -42.0, 777.0},
                  {19.4, 15.0, 20.0}};
    cnn::Matrix expected{{483.3, -178.2, 6509.7},
                         {-1024.84, 733.8, -17680.8},
                         {1136.12, -770.4, 19131.4}};
    
    cnn::Matrix c = a * b;

    EXPECT_TRUE(c == expected);
    EXPECT_DOUBLE_EQ(a(1,1), -13.8);
    EXPECT_DOUBLE_EQ(b(1,1), 15.0);    
}

TEST(Matrix, Multiplication1x1){
    cnn::Matrix a(1,1, 2.0);
    cnn::Matrix b(1,1, 3.0);

    cnn::Matrix c = a * b;

    ASSERT_EQ(c.rows(), 1);
    ASSERT_EQ(c.cols(), 1);
    EXPECT_DOUBLE_EQ(c(0,0), 6.0);
}

TEST(Matrix, MultiplicationByZero){
    cnn::Matrix a(2,2, 2.0);
    cnn::Matrix b(2,2, 0.0);
    cnn::Matrix expected{{0.0, 0.0},
                         {0.0, 0.0}};

    cnn::Matrix c = a * b;

    EXPECT_TRUE(c == expected);
}

TEST(Matrix, MultiplicationOfTheSameMatrices){
    cnn::Matrix a(3,3);
    FillMatrix(a);
    cnn::Matrix b = a;
    cnn::Matrix expected{{30.0, 36.0, 42.0},
                         {66.0, 81.0, 96.0},
                         {102.0, 126.0, 150.0}};

    cnn::Matrix c = a * b;

    EXPECT_TRUE(c == expected);
    EXPECT_DOUBLE_EQ(a(1,1), 5.0);
    EXPECT_DOUBLE_EQ(b(1,1), 5.0);
}

TEST(Matrix, Transposition){
    cnn::Matrix matrix{{1.0, 2.0},
                       {3.0, 4.0},
                       {5.0, 6.0}};
    cnn::Matrix expected{{1.0, 3.0, 5.0},
                         {2.0, 4.0, 6.0}};

    EXPECT_TRUE(matrix.transpose() == expected);
    EXPECT_EQ(matrix.rows(), 3);
    EXPECT_EQ(matrix.cols(), 2);
    EXPECT_DOUBLE_EQ(matrix(1,1), 4.0);
}

TEST(Matrix, DoubleTransposition){
    cnn::Matrix matrix{{1.0, 2.0},
                       {3.0, 4.0},
                       {5.0, 6.0}};

    EXPECT_TRUE(matrix.transpose().transpose() == matrix);
}
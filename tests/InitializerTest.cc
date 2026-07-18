#include <gtest/gtest.h>
#include <cnn/Initializer.h>
#include <cnn/Matrix.h>

TEST(Initializer, Zeros){
    cnn::Matrix expected1{{0.0, 0.0},
                          {0.0, 0.0},
                          {0.0, 0.0}};
    cnn::Matrix matrix1 = cnn::initializer::zeros(3,2);
    EXPECT_FALSE(matrix1.empty());
    EXPECT_TRUE(matrix1 == expected1);

    cnn::Matrix expected2{{0.0, 0.0, 0.0, 0.0},
                          {0.0, 0.0, 0.0, 0.0}};
    cnn::Matrix matrix2 = cnn::initializer::zeros(2,4);
    EXPECT_FALSE(matrix2.empty());
    EXPECT_TRUE(matrix2 == expected2);
}

TEST(Initializer, Ones){
    cnn::Matrix expected1{{1.0, 1.0},
                          {1.0, 1.0},
                          {1.0, 1.0}};
    cnn::Matrix matrix1 = cnn::initializer::ones(3,2);
    EXPECT_FALSE(matrix1.empty());
    EXPECT_TRUE(matrix1 == expected1);

    cnn::Matrix expected2{{1.0, 1.0, 1.0, 1.0},
                          {1.0, 1.0, 1.0, 1.0}};
    cnn::Matrix matrix2 = cnn::initializer::ones(2,4);
    EXPECT_FALSE(matrix2.empty());
    EXPECT_TRUE(matrix2 == expected2);
}

TEST(Initializer, RandomThrowsForMinExceedMax){
    EXPECT_THROW(cnn::initializer::random(2,2,10.0,-1.0), std::invalid_argument);
}

TEST(Initializer, Random){
    cnn::Matrix matrix1 = cnn::initializer::random(1,1,-2.0,2.0);
    EXPECT_FALSE(matrix1.empty());
    EXPECT_EQ(matrix1.rows(), 1);
    EXPECT_EQ(matrix1.cols(), 1);
    EXPECT_GE(matrix1(0,0), -2);
    EXPECT_LE(matrix1(0,0), 2);

    size_t size{100};
    cnn::Matrix matrix2 = cnn::initializer::random(size,size);
    EXPECT_FALSE(matrix2.empty());
    EXPECT_EQ(matrix2.rows(), size);
    EXPECT_EQ(matrix2.cols(), size);
    for(size_t r{0}; r<size; ++r){
        for(size_t c{0}; c<size; ++c){
            EXPECT_GE(matrix2(r,c), -1.0);
            EXPECT_LE(matrix2(r,c), 1.0);
        }
    }
}

TEST(Initializer, EmptyRandom){
    cnn::Matrix matrix1 = cnn::initializer::random(0,0,-2.0,2.0);
    EXPECT_TRUE(matrix1.empty());
    EXPECT_EQ(matrix1.rows(), 0);
    EXPECT_EQ(matrix1.cols(), 0);
}
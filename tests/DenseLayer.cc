#include <gtest/gtest.h>
#include <cnn/DenseLayer.h>

TEST(DenseLayer, setWeightsThrowsForInvalidDimesions){
    cnn::DenseLayer denseLayer(2,2,cnn::activation::relu);
    EXPECT_THROW(denseLayer.setWeights(
                    cnn::Matrix{{1},
                                {0}}), std::invalid_argument);
}

TEST(DenseLayer, setBiasThrowsForInvalidDimesions){
    cnn::DenseLayer denseLayer(2,2,cnn::activation::relu);
    EXPECT_THROW(denseLayer.setBias(
                    cnn::Matrix{{1, 0},
                                {0, 1}}), std::invalid_argument);
}

TEST(DenseLayer, Forward1){
    cnn::Matrix input{{3},
                      {4}};
    cnn::Matrix expected{{3},
                         {4}};

    cnn::DenseLayer denseLayer(2,2,cnn::activation::relu);
    denseLayer.setWeights(cnn::Matrix{{1, 0},
                                      {0, 1}});

    cnn::Matrix result = denseLayer.forward(input);

    EXPECT_TRUE(result == expected);
}

TEST(DenseLayer, Forward2){
    cnn::Matrix input{{3},
                      {4}};
    cnn::Matrix expected{{6},
                         {8}};

    cnn::DenseLayer denseLayer(2,2,cnn::activation::relu);
    denseLayer.setWeights(cnn::Matrix{{2, 0},
                                      {0, 2}});

    cnn::Matrix result = denseLayer.forward(input);

    EXPECT_TRUE(result == expected);
}

TEST(DenseLayer, Forward3){
    cnn::Matrix input{{3},
                      {4}};
    cnn::Matrix expected{{13},
                         {24}};

    cnn::DenseLayer denseLayer(2,2,cnn::activation::relu);
    denseLayer.setWeights(cnn::Matrix{{1, 0},
                                      {0, 1}});
    denseLayer.setBias(cnn::Matrix{{10},
                                   {20}});

    cnn::Matrix result = denseLayer.forward(input);

    EXPECT_TRUE(result == expected);
}

TEST(DenseLayer, Forward4){
    cnn::Matrix input{{-5},
                      {8}};
    cnn::Matrix expected{{0},
                         {8}};

    cnn::DenseLayer denseLayer(2,2,cnn::activation::relu);
    denseLayer.setWeights(cnn::Matrix{{1, 0},
                                      {0, 1}});

    cnn::Matrix result = denseLayer.forward(input);

    EXPECT_TRUE(result == expected);
}
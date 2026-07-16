#include <gtest/gtest.h>
#include <cnn/Activation.h>
#include <cnn/Common.h>

TEST(Activation, Sigmoid){
    EXPECT_LT(cnn::activation::sigmoid(-2), cnn::activation::sigmoid(-1));
    EXPECT_LT(cnn::activation::sigmoid(-1), cnn::activation::sigmoid(0));
    EXPECT_LT(cnn::activation::sigmoid(-0), cnn::activation::sigmoid(1));

    EXPECT_NEAR(cnn::activation::sigmoid(-1000.0), 0.0, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::sigmoid(-6.0), 0.0024726231566347743, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::sigmoid(-2.0), 0.11920292202211755, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::sigmoid(0.0), 0.5, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::sigmoid(1.0), 0.7310585786300049, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::sigmoid(2.0), 0.88079707797788231, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::sigmoid(6.0), 0.99752737684336534, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::sigmoid(1000.0), 1.0, cnn::kEpsilon);
}

TEST(Activation, ReLU){
    EXPECT_DOUBLE_EQ(cnn::activation::relu(-1000.0), 0.0);
    EXPECT_DOUBLE_EQ(cnn::activation::relu(-1e-9), 0.0);
    EXPECT_DOUBLE_EQ(cnn::activation::relu(0.0), 0.0);
    EXPECT_DOUBLE_EQ(cnn::activation::relu(1e-9), 1e-9);
    EXPECT_DOUBLE_EQ(cnn::activation::relu(1.1), 1.1);
    EXPECT_DOUBLE_EQ(cnn::activation::relu(1000.0), 1000.0);
}
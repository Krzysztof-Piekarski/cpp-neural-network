#include <gtest/gtest.h>
#include <cnn/Activation.h>
#include <cnn/Common.h>
#include <cnn/Types.h>

TEST(Activation, Sigmoid){
    EXPECT_LT(cnn::activation::sigmoid(-2), cnn::activation::sigmoid(-1));
    EXPECT_LT(cnn::activation::sigmoid(-1), cnn::activation::sigmoid(0));
    EXPECT_LT(cnn::activation::sigmoid(0), cnn::activation::sigmoid(1));

    EXPECT_NEAR(cnn::activation::sigmoid(-1000.0), 0.0, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::sigmoid(-6.0), 0.0024726231566347743, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::sigmoid(-2.0), 0.11920292202211755, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::sigmoid(0.0), 0.5, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::sigmoid(1.0), 0.7310585786300049, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::sigmoid(2.0), 0.88079707797788231, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::sigmoid(6.0), 0.99752737684336534, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::sigmoid(1000.0), 1.0, cnn::kEpsilon);

    EXPECT_NEAR(cnn::activation::sigmoid(2.4) + cnn::activation::sigmoid(-2.4), 1.0, cnn::kEpsilon);
}

TEST(Activation, SigmoidDerivative){
    EXPECT_LT(cnn::activation::sigmoidDerivative(-1), cnn::activation::sigmoidDerivative(0));
    EXPECT_LT(cnn::activation::sigmoidDerivative(-1e-7), cnn::activation::sigmoidDerivative(0));
    EXPECT_LT(cnn::activation::sigmoidDerivative(1e-7), cnn::activation::sigmoidDerivative(0));
    EXPECT_LT(cnn::activation::sigmoidDerivative(1), cnn::activation::sigmoidDerivative(0));

    EXPECT_NEAR(cnn::activation::sigmoidDerivative(-1000.0), 0.0, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::sigmoidDerivative(-6.0), 0.0024665092913600485, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::sigmoidDerivative(-2.0), 0.10499358540350651, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::sigmoidDerivative(0.0), 0.25, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::sigmoidDerivative(1.0), 0.19661193324148185, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::sigmoidDerivative(2.0), 0.10499358540350662, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::sigmoidDerivative(6.0), 0.0024665092913599309, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::sigmoidDerivative(1000.0), 0.0, cnn::kEpsilon);

    EXPECT_NEAR(cnn::activation::sigmoidDerivative(2.4), cnn::activation::sigmoidDerivative(-2.4), cnn::kEpsilon);
}

TEST(Activation, ReLU){
    EXPECT_DOUBLE_EQ(cnn::activation::relu(-1000.0), 0.0);
    EXPECT_DOUBLE_EQ(cnn::activation::relu(-1e-9), 0.0);
    EXPECT_DOUBLE_EQ(cnn::activation::relu(0.0), 0.0);
    EXPECT_DOUBLE_EQ(cnn::activation::relu(1e-9), 1e-9);
    EXPECT_DOUBLE_EQ(cnn::activation::relu(1.1), 1.1);
    EXPECT_DOUBLE_EQ(cnn::activation::relu(1000.0), 1000.0);
}

TEST(Activation, ReLUDerivative){
    EXPECT_DOUBLE_EQ(cnn::activation::reluDerivative(-1e-9), 0.0);
    EXPECT_DOUBLE_EQ(cnn::activation::reluDerivative(0.0), 0.0);
    EXPECT_DOUBLE_EQ(cnn::activation::reluDerivative(1e-9), 1.0);
    EXPECT_DOUBLE_EQ(cnn::activation::reluDerivative(1.1), 1.0);
}

TEST(Activation, Tanh){
    EXPECT_NEAR(cnn::activation::tanh(-1000.0), -1.0, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::tanh(-1.1), -0.8004990217606297, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::tanh(0.0), 0.0, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::tanh(1.1), 0.8004990217606297, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::tanh(1000.0), 1.0, cnn::kEpsilon);

    EXPECT_NEAR(cnn::activation::tanh(-2.4) + cnn::activation::tanh(2.4), 0.0, cnn::kEpsilon);
}

TEST(Activation, TanhDerivative){
    EXPECT_NEAR(cnn::activation::tanhDerivative(-1000.0), 0.0, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::tanhDerivative(-1.1), 0.35920131616027484, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::tanhDerivative(0.0), 1.0, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::tanhDerivative(1.1), 0.35920131616027484, cnn::kEpsilon);
    EXPECT_NEAR(cnn::activation::tanhDerivative(1000.0), 0.0, cnn::kEpsilon);

    EXPECT_NEAR(cnn::activation::tanhDerivative(2.4), cnn::activation::tanhDerivative(-2.4), cnn::kEpsilon);
}

TEST(Activation, ApplyThrowsForUnknownActivation){
    cnn::ActivationType invalidType = static_cast<cnn::ActivationType>(999);

    EXPECT_THROW(
        cnn::activation::apply(1.0, invalidType), std::invalid_argument
    );
}

TEST(Activation, DerivativeThrowsForUnknownActivation){
    cnn::ActivationType invalidType = static_cast<cnn::ActivationType>(999);

    EXPECT_THROW(
        cnn::activation::derivative(1.0, invalidType), std::invalid_argument
    );
}

TEST(Activation, ApplyUsesCorrectFunction){
    EXPECT_NEAR(cnn::activation::apply(0.0, cnn::ActivationType::Sigmoid),
                0.5,
                cnn::kEpsilon);

    EXPECT_DOUBLE_EQ(cnn::activation::apply(-1.0, cnn::ActivationType::Relu),
                0.0);

    EXPECT_NEAR(cnn::activation::apply(1.0, cnn::ActivationType::Tanh),
                cnn::activation::tanh(1.0),
                cnn::kEpsilon);
}

TEST(Activation, DerivativeUsesCorrectFunction){
    EXPECT_NEAR(cnn::activation::derivative(0.0, cnn::ActivationType::Sigmoid),
                0.25,
                cnn::kEpsilon);

    EXPECT_DOUBLE_EQ(cnn::activation::derivative(-1.0, cnn::ActivationType::Relu),
                0.0);

    EXPECT_NEAR(cnn::activation::derivative(0.0, cnn::ActivationType::Tanh),
                1.0,
                cnn::kEpsilon);
}
#include <gtest/gtest.h>
#include <cnn/Loss.h>
#include <cnn/Common.h>

TEST(Loss, MeanSquaredErrorThrowsForIncorrectDimensions){
    cnn::Matrix prediction{{2}, {4}, {2}};
    cnn::Matrix target{{1}, {2}, {3}, {4}};

    EXPECT_THROW(cnn::loss::meanSquaredError(prediction, target), std::invalid_argument);
}

TEST(Loss, MeanSquaredErrorDoesntEditPredictionAndTarget){
    cnn::Matrix prediction{{2}, {4}, {2}};
    cnn::Matrix target{{1}, {2}, {3}};

    EXPECT_NEAR(cnn::loss::meanSquaredError(prediction, target), 2.0, cnn::kEpsilon);

    EXPECT_TRUE((prediction == cnn::Matrix{{2}, {4}, {2}}));
    EXPECT_TRUE((target == cnn::Matrix{{1}, {2}, {3}}));
}

TEST(Loss, MeanSquaredError){
    cnn::Matrix prediction{{1.0, 2.0},
                           {3.0, 1.0}};
    cnn::Matrix target{{0.9, 2.2},
                       {2.8, 1.2}};

    EXPECT_NEAR(cnn::loss::meanSquaredError(prediction, target), 0.0325, cnn::kEpsilon);
}

TEST(Loss, MeanSquaredErrorForZeroError){
    cnn::Matrix prediction{{1}, {2}, {3}};
    cnn::Matrix target{{1}, {2}, {3}};

    EXPECT_NEAR(cnn::loss::meanSquaredError(prediction, target), 0.0, cnn::kEpsilon);
}

TEST(Loss, MeanSquaredErrorDerivativeThrowsForIncorrectDimensions){
    cnn::Matrix prediction{{2}, {4}, {2}};
    cnn::Matrix target{{1}, {2}, {3}, {4}};

    EXPECT_THROW(cnn::loss::meanSquaredErrorDerivative(prediction, target), std::invalid_argument);
}

TEST(Loss, MeanSquaredErrorDerivativeDoesntEditPredictionAndTarget){
    cnn::Matrix prediction{{2}, {4}, {2}, {4}};
    cnn::Matrix target{{1}, {2}, {3}, {4}};

    EXPECT_TRUE((cnn::loss::meanSquaredErrorDerivative(prediction, target) ==
                cnn::Matrix{{0.5}, {1.0}, {-0.5}, {0.0}}));

    EXPECT_TRUE((prediction == cnn::Matrix{{2}, {4}, {2}, {4}}));
    EXPECT_TRUE((target == cnn::Matrix{{1}, {2}, {3}, {4}}));
}

TEST(Loss, MeanSquaredErrorDerivative){
    cnn::Matrix prediction{{2}, {4}, {6}};
    cnn::Matrix target{{1}, {2}, {3}};

    EXPECT_TRUE((cnn::loss::meanSquaredErrorDerivative(prediction, target) ==
                cnn::Matrix{{2.0/3.0}, {4.0/3}, {2.0}}));
}
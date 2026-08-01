#include <gtest/gtest.h>
#include <cnn/DenseLayer.h>
#include <cnn/Common.h>

TEST(DenseLayer, ConstructorThrowsForInvalidDimensions){
    EXPECT_THROW(cnn::DenseLayer(0, 4), std::invalid_argument);

    EXPECT_THROW(cnn::DenseLayer(3, 0), std::invalid_argument);
}

TEST(DenseLayer, ConstructorCreatesCorrectDimensions){
    cnn::DenseLayer layer(3, 7);

    EXPECT_EQ(layer.weights().rows(), 7);
    EXPECT_EQ(layer.weights().cols(), 3);

    EXPECT_EQ(layer.bias().rows(), 7);
    EXPECT_EQ(layer.bias().cols(), 1);

    for(size_t r{0}; r<layer.weights().rows(); ++r){
        for(size_t c{0}; c<layer.weights().cols(); ++c){
            EXPECT_GE(layer.weights()(r,c), -1.0);
            EXPECT_LE(layer.weights()(r,c), 1.0);
        }
    }
}

TEST(DenseLayer, WeightsReturnsReference){
    cnn::DenseLayer layer(2, 2);

    const auto& weights = layer.weights();

    EXPECT_EQ(&weights, &layer.weights());
}

TEST(DenseLayer, BiasReturnsReference){
    cnn::DenseLayer layer(2, 2);

    const auto& bias = layer.bias();

    EXPECT_EQ(&bias, &layer.bias());
}

TEST(DenseLayer, SetWeightsThrowsForInvalidDimensions){
    cnn::DenseLayer layer(2, 2, cnn::ActivationType::Relu);

    EXPECT_THROW(layer.setWeights(
                    cnn::Matrix{{1},
                                {0}}), std::invalid_argument);
}

TEST(DenseLayer, SetWeights){
    cnn::DenseLayer layer(2, 2);

    cnn::Matrix weights{{1, 2},
                        {3, 4}};

    layer.setWeights(weights);

    EXPECT_TRUE(layer.weights() == weights);
}

TEST(DenseLayer, SetBiasThrowsForInvalidDimensions){
    cnn::DenseLayer layer(2, 2, cnn::ActivationType::Relu);

    EXPECT_THROW(layer.setBias(
                    cnn::Matrix{{1, 0},
                                {0, 1}}), std::invalid_argument);
}

TEST(DenseLayer, SetBias){
    cnn::DenseLayer layer(2,2);

    cnn::Matrix bias{{1},
                     {2}};

    layer.setBias(bias);

    EXPECT_TRUE(layer.bias() == bias);
}

TEST(DenseLayer, ForwardThrowsForInvalidInputDimensions){
    cnn::DenseLayer layer(2, 2, cnn::ActivationType::Relu);

    EXPECT_THROW(layer.forward(
                    cnn::Matrix{{1},
                                {2},
                                {3}}), std::invalid_argument);
}

TEST(DenseLayer, ForwardUsesSigmoidByDefault){
    cnn::DenseLayer layer(1, 1);

    layer.setWeights(cnn::Matrix{{1}});
    layer.setBias(cnn::Matrix{{0}});

    cnn::Matrix result = layer.forward(cnn::Matrix{{0}});

    EXPECT_NEAR(result(0,0), 0.5, cnn::kEpsilon);
}

TEST(DenseLayer, ForwardIdentityWeights){
    cnn::Matrix input{{3},
                      {4}};
    cnn::Matrix expected{{3},
                         {4}};

    cnn::DenseLayer layer(2, 2, cnn::ActivationType::Relu);
    layer.setWeights(cnn::Matrix{{1, 0},
                                 {0, 1}});

    cnn::Matrix result = layer.forward(input);

    EXPECT_TRUE(result == expected);
}

TEST(DenseLayer, ForwardScalesInput){
    cnn::Matrix input{{3},
                      {4}};
    cnn::Matrix expected{{6},
                         {8}};

    cnn::DenseLayer layer(2, 2, cnn::ActivationType::Relu);
    layer.setWeights(cnn::Matrix{{2, 0},
                                 {0, 2}});

    cnn::Matrix result = layer.forward(input);

    EXPECT_TRUE(result == expected);
}

TEST(DenseLayer, ForwardAddsBias){
    cnn::Matrix input{{3},
                      {4}};
    cnn::Matrix expected{{13},
                         {24}};

    cnn::DenseLayer layer(2, 2, cnn::ActivationType::Relu);
    layer.setWeights(cnn::Matrix{{1, 0},
                                 {0, 1}});
    layer.setBias(cnn::Matrix{{10},
                              {20}});

    cnn::Matrix result = layer.forward(input);

    EXPECT_TRUE(result == expected);
}

TEST(DenseLayer, ForwardAppliesRelu){
    cnn::Matrix input{{-5},
                      {8}};
    cnn::Matrix expected{{0},
                         {8}};

    cnn::DenseLayer layer(2, 2, cnn::ActivationType::Relu);
    layer.setWeights(cnn::Matrix{{1, 0},
                                 {0, 1}});

    cnn::Matrix result = layer.forward(input);

    EXPECT_TRUE(result == expected);
}

TEST(DenseLayer, ForwardAppliesActivationAfterBias){
    cnn::Matrix input{{4},
                      {8}};
    cnn::Matrix expected{{10},
                         {0}};

    cnn::DenseLayer layer(2, 2, cnn::ActivationType::Relu);
    layer.setWeights(cnn::Matrix{{2, 0},
                                 {0, 2}});
    layer.setBias(cnn::Matrix{{2},
                              {-17}});

    cnn::Matrix result = layer.forward(input);

    EXPECT_TRUE(result == expected);
}

TEST(DenseLayer, ForwardTrainingThrowsForInvalidInputDimensions){
    cnn::DenseLayer layer(2, 2, cnn::ActivationType::Relu);

    EXPECT_THROW(layer.forwardTraining(
                    cnn::Matrix{{1},
                                {2},
                                {3}}), std::invalid_argument);
}

TEST(DenseLayer, ForwardTrainingUsesSigmoidByDefault){
    cnn::DenseLayer layer(1, 1);

    layer.setWeights(cnn::Matrix{{1}});
    layer.setBias(cnn::Matrix{{0}});

    cnn::Matrix result = layer.forwardTraining(cnn::Matrix{{0}});

    EXPECT_NEAR(result(0,0), 0.5, cnn::kEpsilon);
}

TEST(DenseLayer, ForwardTrainingIdentityWeights){
    cnn::Matrix input{{3},
                      {4}};
    cnn::Matrix expected{{3},
                         {4}};

    cnn::DenseLayer layer(2, 2, cnn::ActivationType::Relu);
    layer.setWeights(cnn::Matrix{{1, 0},
                                 {0, 1}});

    cnn::Matrix result = layer.forwardTraining(input);

    EXPECT_TRUE(result == expected);
}

TEST(DenseLayer, ForwardTrainingScalesInput){
    cnn::Matrix input{{3},
                      {4}};
    cnn::Matrix expected{{6},
                         {8}};

    cnn::DenseLayer layer(2, 2, cnn::ActivationType::Relu);
    layer.setWeights(cnn::Matrix{{2, 0},
                                 {0, 2}});

    cnn::Matrix result = layer.forwardTraining(input);

    EXPECT_TRUE(result == expected);
}

TEST(DenseLayer, ForwardTrainingAddsBias){
    cnn::Matrix input{{3},
                      {4}};
    cnn::Matrix expected{{13},
                         {24}};

    cnn::DenseLayer layer(2, 2, cnn::ActivationType::Relu);
    layer.setWeights(cnn::Matrix{{1, 0},
                                 {0, 1}});
    layer.setBias(cnn::Matrix{{10},
                              {20}});

    cnn::Matrix result = layer.forwardTraining(input);

    EXPECT_TRUE(result == expected);
}

TEST(DenseLayer, ForwardTrainingAppliesRelu){
    cnn::Matrix input{{-5},
                      {8}};
    cnn::Matrix expected{{0},
                         {8}};

    cnn::DenseLayer layer(2, 2, cnn::ActivationType::Relu);
    layer.setWeights(cnn::Matrix{{1, 0},
                                 {0, 1}});

    cnn::Matrix result = layer.forwardTraining(input);

    EXPECT_TRUE(result == expected);
}

TEST(DenseLayer, ForwardTrainingAppliesActivationAfterBias){
    cnn::Matrix input{{4},
                      {8}};
    cnn::Matrix expected{{10},
                         {0}};

    cnn::DenseLayer layer(2, 2, cnn::ActivationType::Relu);
    layer.setWeights(cnn::Matrix{{2, 0},
                                 {0, 2}});
    layer.setBias(cnn::Matrix{{2},
                              {-17}});

    cnn::Matrix result = layer.forwardTraining(input);

    EXPECT_TRUE(result == expected);
}

TEST(DenseLayer, CallBackwardBeforeForwardTrainingThrows){
    cnn::DenseLayer layer(2, 2);

    EXPECT_THROW(layer.backward(
                        cnn::Matrix{{1.0},
                                    {2.0}}), std::logic_error);
}

TEST(DenseLayer, BackwardThrowsForInvalidGradientDimensions){
    cnn::DenseLayer layer(2, 2);

    layer.forwardTraining(cnn::Matrix{{1.0},
                                      {2.0}});

    EXPECT_THROW(layer.backward(
                        cnn::Matrix{{1.0},
                                    {2.0},
                                    {3.0}}), std::invalid_argument);
}

TEST(DenseLayer, BackwardReturnsInputGradient){
    cnn::DenseLayer layer(2, 2, cnn::ActivationType::Relu);
    layer.setWeights(cnn::Matrix{{1.0, 0.0},
                                 {0.0, 3.0}});
    cnn::Matrix input{{3.0},
                      {4.0}};

    layer.forwardTraining(input);

    cnn::Matrix gradient{{1.0},
                         {1.0}};

    cnn::Matrix result = layer.backward(gradient);

    EXPECT_TRUE((result == cnn::Matrix{{1.0},
                                       {3.0}}));
}

TEST(DenseLayer, CallUpdateParametersBeforeBackwardThrows){
    cnn::DenseLayer layer(2, 2);

    EXPECT_THROW(layer.updateParameters(1.0), std::logic_error);
}

TEST(DenseLayer, UpdateParametersThrowsForInvalidLearningRate){
    cnn::DenseLayer layer(2, 2);
    cnn::Matrix input{{3.0},
                      {4.0}};

    layer.forwardTraining(input);

    cnn::Matrix gradient{{1.0},
                         {1.0}};

    layer.backward(gradient);

    EXPECT_THROW(layer.updateParameters(0.0), std::invalid_argument);
    EXPECT_THROW(layer.updateParameters(-0.01), std::invalid_argument);
}

TEST(DenseLayer, UpdateParametersResetsGradients){
    cnn::DenseLayer layer(2, 2);
    cnn::Matrix input{{3.0},
                      {4.0}};

    layer.forwardTraining(input);

    cnn::Matrix gradient{{1.0},
                         {1.0}};

    layer.backward(gradient);

    layer.updateParameters(0.1);

    EXPECT_THROW(layer.updateParameters(0.1), std::logic_error);
}

TEST(DenseLayer, UpdateParameters){
    cnn::DenseLayer layer(2, 2, cnn::ActivationType::Relu);
    cnn::Matrix expectedWeightsAfterUpdate{{0.7, 1.4},
                                           {2.7, 3.4}};
    cnn::Matrix expectedBiasesAfterUpdate{{4.7},
                                          {5.7}};

    layer.setWeights(cnn::Matrix{{1.0, 2.0},
                                 {3.0, 4.0}});

    layer.setBias(cnn::Matrix{{5.0},
                              {6.0}});

    layer.forwardTraining(cnn::Matrix{{1.0},
                                      {2.0}});

    layer.backward(cnn::Matrix{{1.0},
                               {1.0}});

    layer.updateParameters(0.3);
    
    EXPECT_TRUE(layer.weights() == expectedWeightsAfterUpdate);
    EXPECT_TRUE(layer.bias() == expectedBiasesAfterUpdate);
}

TEST(DenseLayer, CallCalculateOutputDeltaBeforeForwardTrainingThrows){
    cnn::DenseLayer layer(2, 2);

    EXPECT_THROW(layer.calculateOutputDelta(
                        cnn::Matrix{{1.0},
                                    {2.0}}), std::logic_error);
}

TEST(DenseLayer, CalculateOutputDeltaThrowsForInvalidDimensions){
    cnn::DenseLayer layer(2, 2);

    layer.forwardTraining(cnn::Matrix{{1.0},
                                      {2.0}});


    EXPECT_THROW(layer.calculateOutputDelta(
                        cnn::Matrix{{1.0},
                                    {2.0},
                                    {3.0}}), std::invalid_argument);
}

TEST(DenseLayer, CalculateOutputDeltaReluPositive){
    cnn::DenseLayer layer(1,1, cnn::ActivationType::Relu);
    layer.setWeights(cnn::Matrix{{1.0}});

    layer.forwardTraining(cnn::Matrix{{3.0}});

    cnn::Matrix result = layer.calculateOutputDelta(cnn::Matrix{{7.0}});
    EXPECT_TRUE((result == cnn::Matrix{{7.0}}));
}

TEST(DenseLayer, CalculateOutputDeltaReluNegative){
    cnn::DenseLayer layer(1,1, cnn::ActivationType::Relu);
    layer.setWeights(cnn::Matrix{{1.0}});

    layer.forwardTraining(cnn::Matrix{{-3.0}});

    cnn::Matrix result = layer.calculateOutputDelta(cnn::Matrix{{7.0}});
    EXPECT_TRUE((result == cnn::Matrix{{0.0}}));
}

TEST(DenseLayer, CalculateOutputDeltaDefaultSigmoid){
    cnn::DenseLayer layer(1,1);
    layer.setWeights(cnn::Matrix{{1.0}});

    layer.forwardTraining(cnn::Matrix{{0}});

    cnn::Matrix result = layer.calculateOutputDelta(cnn::Matrix{{8.0}});
    EXPECT_TRUE((result == cnn::Matrix{{2.0}}));
}

TEST(DenseLayer, CalculateOutputDeltaTanh){
    cnn::DenseLayer layer(1,1, cnn::ActivationType::Tanh);
    layer.setWeights(cnn::Matrix{{1.0}});

    layer.forwardTraining(cnn::Matrix{{0.0}});

    cnn::Matrix result = layer.calculateOutputDelta(cnn::Matrix{{4.0}});
    EXPECT_TRUE((result == cnn::Matrix{{4.0}}));
}

TEST(DenseLayer, CalculateOutputReluManyNeurons){
    cnn::DenseLayer layer(3,3, cnn::ActivationType::Relu);
    layer.setWeights(cnn::Matrix{{1, 0, 0},
                                 {0, 1, 0},
                                 {0, 0, 1}});

    layer.forwardTraining(cnn::Matrix{{-2.0},
                                      {3.0},
                                      {-5.0}});

    cnn::Matrix result = layer.calculateOutputDelta(cnn::Matrix{{5.0},
                                                                {7.0},
                                                                {9.0}});
    EXPECT_TRUE((result == cnn::Matrix{{0.0},
                                       {7.0},
                                       {0.0}}));
}

TEST(DenseLayer, InputSize){
    cnn::DenseLayer layer(3, 7);

    EXPECT_EQ(layer.inputSize(), 3);
}

TEST(DenseLayer, OutputSize){
    cnn::DenseLayer layer(3, 7);

    EXPECT_EQ(layer.outputSize(), 7);
}
#include <gtest/gtest.h>
#include <cnn/NeuralNetwork.h>
#include <cnn/Types.h>

TEST(NeuralNetwork, AddLayerThrowsForInputSizeDoesNotMatchPreviousLayerOutput){
    cnn::NeuralNetwork net;
    cnn::DenseLayer layer(2,3);
    net.addLayer(layer);

    EXPECT_THROW(net.addLayer(cnn::DenseLayer(7,21)), std::invalid_argument);
}

TEST(NeuralNetwork, AddLayerAcceptsMatchingLayers){
    cnn::NeuralNetwork net;
    cnn::DenseLayer layer(2,3);
    net.addLayer(layer);

    EXPECT_NO_THROW(net.addLayer(cnn::DenseLayer(3,8)));
    EXPECT_NO_THROW(net.addLayer(cnn::DenseLayer(8,1)));
    EXPECT_NO_THROW(net.addLayer(cnn::DenseLayer(1,20)));
    EXPECT_NO_THROW(net.addLayer(cnn::DenseLayer(20,2)));
}

TEST(NeuralNetwork, AddLayerStoresLayer){
    cnn::NeuralNetwork net;
    
    cnn::DenseLayer layer(2,1, cnn::ActivationType::Relu);
    layer.setWeights({{5,8}});

    net.addLayer(layer);

    layer.setWeights({{1,1}});

    EXPECT_TRUE(net.predict(cnn::Matrix{{1},{1}}) == cnn::Matrix{{13.0}});
}

TEST(NeuralNetwork, EmptyNeuralNetworkThrowsForPredict){
    cnn::NeuralNetwork net;

    EXPECT_THROW(net.predict(cnn::Matrix{{1}, {2}}), std::logic_error);
}

TEST(NeuralNetwork, PredictThrowsForNotMatchingInputSize){
    cnn::NeuralNetwork net;
    cnn::Matrix input{{-1}, {2}};
    cnn::DenseLayer layer(3,2);

    net.addLayer(layer);

    EXPECT_THROW(net.predict(input), std::invalid_argument);
}

TEST(NeuralNetwork, PredictAcceptsMatchingInputSize){
    cnn::NeuralNetwork net;
    cnn::Matrix input{{-1}, {2}, {3}};
    cnn::DenseLayer layer(3,2);

    net.addLayer(layer);

    EXPECT_NO_THROW((net.predict(input)));
}

TEST(NeuralNetwork, PredictSingleLayer){
    cnn::NeuralNetwork net;
    cnn::Matrix input{{-1}, {2}, {0}};
    cnn::DenseLayer layer(3,3);

    net.addLayer(layer);

    EXPECT_TRUE(net.predict(input) == layer.forward(input));
}

TEST(NeuralNetwork, PredictTwoLayers){
    cnn::NeuralNetwork net;
    cnn::DenseLayer layer1(2,2, cnn::ActivationType::Relu);
    cnn::DenseLayer layer2(2,1, cnn::ActivationType::Relu);

    layer1.setWeights({{1, 0},
                       {0, 1}});
    layer2.setWeights({{2,3}});

    net.addLayer(layer1);
    net.addLayer(layer2);

    EXPECT_TRUE(net.predict(cnn::Matrix{{1.0},
                                        {2.0}}) == cnn::Matrix{{8.0}});
}

TEST(NeuralNetwork, PredictTwoLayersWithDifferentActivationFunction){
    cnn::NeuralNetwork net;
    cnn::DenseLayer layer1(2,2, cnn::ActivationType::Relu);
    cnn::DenseLayer layer2(2,1, cnn::ActivationType::Sigmoid);

    layer1.setWeights({{1, 0},
                       {0, 1}});
    layer2.setWeights({{2,3}});

    net.addLayer(layer1);
    net.addLayer(layer2);

    cnn::Matrix output = net.predict(cnn::Matrix{{1.0},
                                                 {2.0}});

    EXPECT_TRUE(output == cnn::Matrix{{cnn::activation::apply(8.0,
                                       cnn::ActivationType::Sigmoid)}});
}

TEST(NeuralNetwork, PredictTwoLayersWithBiases){
    cnn::NeuralNetwork net;
    cnn::DenseLayer layer1(2,2, cnn::ActivationType::Relu);
    cnn::DenseLayer layer2(2,1, cnn::ActivationType::Relu);

    layer1.setWeights({{1, 0},
                       {0, 1}});
    layer2.setWeights({{2,3}});

    layer1.setBias({{1}, {1}});
    layer2.setBias({{2}});

    net.addLayer(layer1);
    net.addLayer(layer2);

    EXPECT_TRUE(net.predict(cnn::Matrix{{1.0},
                                        {2.0}}) == cnn::Matrix{{15.0}});
}

TEST(NeuralNetwork, PredictIsDeterministic){
    cnn::NeuralNetwork net;
    cnn::DenseLayer layer1(2,2, cnn::ActivationType::Relu);
    cnn::DenseLayer layer2(2,1, cnn::ActivationType::Relu);
    cnn::Matrix input{{1.0},
                      {2.0}};

    layer1.setWeights({{1, 0},
                       {0, 1}});
    layer2.setWeights({{2,3}});

    net.addLayer(layer1);
    net.addLayer(layer2);

    cnn::Matrix output1 = net.predict(input);
    cnn::Matrix output2 = net.predict(input);
    cnn::Matrix output3 = net.predict(input);
    cnn::Matrix output4 = net.predict(input);

    EXPECT_TRUE(output1 == output2 &&
                output2 == output3 &&
                output3 == output4);
}
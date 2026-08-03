# cpp-neural-network

A neural network library written from scratch in modern C++.

## Features

- Matrix implementation
- Dense layers
- ReLU, Sigmoid and Tanh activations
- Mean Squared Error loss
- Forward propagation
- Backpropagation
- Gradient descent training
- Unit tests (GoogleTest)

## Goals

- Learn how neural networks work from first principles.
- Implement forward propagation.
- Implement backpropagation.
- Train the network on XOR and MNIST.
- Keep the implementation lightweight and dependency-free where possible.

## Build
```bash
cmake --preset default
cmake --build --preset default
```

## Run tests
```bash
ctest --preset default
```

## Examples

After building, the example programs are located in:

```bash
./build/examples/
```
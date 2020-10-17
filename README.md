# nn++
A header-only library for creating neural network models in C++

## Features
* **Activation functions**
    * Linear
    * ReLU
    * Leaky ReLU
    * Sigmoid
    * Tanh
    * Softmax
* **Loss functions**
    * Mean squared error
    * Binary cross-entropy
    * Categorical cross-entropy
* **Optimizers**
    * Stochastic gradient descent
    * Momentum
    * RMSProp
    * Adam

nn++ currently only supports densely-connected feedforward neural networks.

## Examples
* [addition.h](nn++/source/examples/addition/addition.h): Learning the addition operation
* [mnist.h](nn++/source/examples/mnist/mnist.h): Learning to classify hand-written digits

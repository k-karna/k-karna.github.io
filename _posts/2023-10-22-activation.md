---
title: Activation Functions, Optimization Methods, and Loss Functions
tags: deep-learning
sidebar:
  nav: "docs-en"
mathjax: true
mathjax_autoNumber: true
---

## Activation Functions
In its simplest form, the neuron output of each layer is computed as: $a_{n+1} = w^T z_n + b_n$ where $w_n$ and $b_n$ are weight and bias parameters at layer $n$, respectively, and $z_n$ is neuron output of previous layer $n_1$ computed by a differentiable non-linear function $f(\cdot):z_n = f(a_n)$. This fixed non-linear function is known as __activation function__ (Apicella _et al.,_ 2021).

### Sigmoid, Hard-Sigmoid
The most common activation function is __Sigmoid__, also known as logistic. It is a bounded differentiable real-function defined as: 

$$\text{Sigmoid} \quad or \quad σ = \frac{1}{1 + e^{-x}}$$

Major problem with sigmoid is that, it binds all inputs between $0$ and $1$, where a large change of inputs leads to small change in output, resulting in smaller gradient values. When network is trained over many layers, these smaller gradient creates a __vanishing gradient problem__. 

As a solution with Sigmoid, we have a __Hard-Sigmoid__ which introduce linear behavior around $0$ to allow gradient flow easily. It can be defined as:

$$\text{Hard-Sigmoid} = \text{max}(\text{min}(\sigma, 1),0)$$

### SoftSign

Softsign activation function is similar to Sigmoid (having "S"-shaped curve). It is also continuous, differentiable and can be defined as :

$$ \text{SoftSign}(x) = \frac{x}{\lvert x \rvert + 1} $$

If input is positive, SoftSign bind output between $0$ and $1$. However, it binds between $-1$ and $0$ for negative inputs.

### TanH, Hard-TanH

Hyperbolic Tangent (TanH) is similar to Sigmoid, continuous, bounded, differentiable and defined as:

$$ \text{tanh} = \frac{1 - e^{-x}}{1 + e^{x}}$$

It has improved range of output i.e., between $-1$ to $1$. However, problem of large change of inputs leading to smaller change in output is not resolved, even with __Hard-Tanh__ which can be expressed as: 

$$\text{Hard-tanH} = \text{max}(\text{min}(\text{tanh}, 1),-1)$$

### ReLU

Rectified Linear Unit (ReLU) is continuous, non-bounded and unlike Sigmoid and Tanh, not-zero centered activation function that can be written as:

$$f(x) = \text{max}(0,x)$$

It is not exponential, so computationally cheap, and __alleviates the vanishing gradient problem__ for being not bounded in at least one direction. However, as negative inputs to ReLU evaluates to $0$, it start to create a problem to __dead neuron__

### Leaky-ReLU, PReLU

__Leaky-ReLU (LReLU)__

It attempts to solve __dead neuron__ issue with ReLU by allowing small gradient to flow when inputs are non-positive. It can defined as: 

$$
\text{LReLU}(x) =
\begin{cases}
x,  & \text{if $x \ge 0$} \\
0.01 ⋅ x, & \text{otherwise}
\end{cases}$$

However, it does not bring significant improvement, rather possibility of __vanishing gradient problem__ coming back.

__Parametric ReLU (PReLU)__:

PReLU attempts to resolve Leaky-ReLU problem by taking additional parameter $\alpha$. This additional parameter is learnt jointly with whole model using gradient method without weight decay(to not push $\alpha$ to zero). It can be defined as:

$$
\text{PReLU}(x) =
\begin{cases}
x,  & \text{if $x \ge 0$} \\
\alpha ⋅ x, & \text{otherwise}
\end{cases}$$

It is not computationally expensive to ReLU or Leaky-ReLU and slightly improves on __vanishing gradient__

### Softplus

Another activation function similar to ReLU is __softplus__ It is smooth approximation of ReLU function and defined as: 

$$\text{softplus}(x) = \log(1 + \text{exp}(x))$$

This was proposed to outperform ReLU, however results are more or less similar, with softplus being computationally costly.

### Exponentially Linear Units (ELU)

It is another method similar to ReLU (or parametric ReLU). It can be defined as: 

$$
\text{ELU}(x) =
\begin{cases}
x,  & \text{if $x \ge 0$} \\
\alpha ⋅ (\text{exp}(x) -1), & \text{otherwise}
\end{cases}$$

With the additional parameter $\alpha$ controlling the values for negative inputs, ELU allows faster learning as values given by ELU units push the mean of activation closer to $0$.

### Gated Linear Units (GLU)
### Swish Activation
### Mish Activation

### Softmax

Softmax activation function is used in the final layer of network for multi-class classification tasks. It maps output as a probability distribution in the range of $[0,1]$ and sum of each outcome is equal to $1$. Given a input vector $\overrightarrow z $ with $K$ classes, softmax can be defined as:

$$\text{softmax}(\overrightarrow z) = \frac{e^{z_{i}}}{\sum_{j =1}^K e^{z_{i}}} $$

## Optimization Methods
### Gradient Descent
### Mini-Batch Gradient Descent
### ADAM
### RMSProp
### AdaGrad
### Adadelta
### Nesterov Acc. Gradient (NAG)
### L-BFGS

## Loss Functions

## References
- Apicella, A., Donnarumma, F., Isgrò, F. and Prevete, R., 2021. A survey on modern trainable activation functions. _Neural Networks_, 138, pp.14-32.
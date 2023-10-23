---
title: Activation Functions, Optimization Methods, and Loss Functions
tags: northumbria-article
sidebar:
  nav: "docs-en"
mathjax: true
---

## Activation Functions
In its simplest form, the neuron output of each layer is computed as: $a_{n+1} = w^T z_n + b_n$ where $w_n$ and $b_n$ are weight and bias parameters at layer $n$, respectively, and $z_n$ is neuron output of previous layer $n_1$ computed by a differentiable non-linear function $f(\cdot):z_n = f(a_n)$. This fixed non-linear function is known as __activation function__ (Apicella _et al.,_ 2021)
### Sigmoid 
### TanH
### ReLU, LeakyReLU
### Exponentially Linear Units (ELU)
### Gated Linear UNits (GLU)
### Swish Activation
### Mish Activation

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
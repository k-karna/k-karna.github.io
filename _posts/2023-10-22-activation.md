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

### Exponential Linear Units (ELU), PELU, SELU

It is another method similar to ReLU (or parametric ReLU). It can be defined as: 

$$
\text{ELU}(x) =
\begin{cases}
x,  & \text{if $x \ge 0$} \\
\alpha ⋅ (\text{exp}(x) -1), & \text{otherwise}
\end{cases}$$

With the additional parameter $\alpha$ controlling the values for negative inputs, ELU allows faster learning as values given by ELU units push the mean of activation closer to $0$.

__Parametric Exponential Linear Units (PELU)__

PELU takes two trainable parameter, that do not need to be manually set, learned with other network parameters using gradient method. It can be defined as :

$$
\text{PELU}(x) =
\begin{cases}
\frac{\beta}{\gamma}x,  & \text{if $x \ge 0$} \\
\beta ⋅ (\text{exp}(\frac{x}{\gamma}) -1), & \text{otherwise}
\end{cases}$$

__Scaled Exponential Linear Units (SELU)__

SELU has additional scaling hyper-parameter $\lambda$. It can be defined as: 

$$
\text{SELU}(x) = λ 
\begin{cases}
x,  & \text{if $x \ge 0$} \\
\alpha ⋅ (\text{exp}^x -1), & \text{otherwise}
\end{cases}
$$


Here,  $λ ≈ 1.05070098$, and $α ≈ 1.67326324$. SELU is effective when it comes to covariate shift, and vanishing / exploding gradient problem for having __self-normalizing__ property. By self-normalizing, we mean if the SELU inputs follows a Gaussian distribution with mean and variance around $0$ and $1$, respectively, the mean and variance of SELU are also around $0$ and $1$. 

### SiLU

__Sigmoid-weighted Linear Units (SiLU)__

SiLU is sigmoid function weighted by its inputs, so can be expressed as:

$$ \text{SiLU}(x) = x ⋅ \text{sigmoid}(x)$$

### Swish Activation, E-Swish

__Swish__ is another function based on Sigmoid, but similar to ReLU for being unbounded above and bounded below. However, unlike ReLU, Swish is non-monotonic, and smooth. It can be defined as:

$$ \text{Swish}(x) = x ⋅ \text{Sigmoid}{β ⋅ x}$$

They key thing with Swish is when trainable parameter $\beta$ approaches $\infty$, it behaves like ReLU, and when $β = 1$, it is similar to SiLU. (Ramchandran, Zoph & Le, 2017)

__E-Swish__

E-Swish is similar to SiLU, but with additional parameter that needs to be tuned by user. It can be written as:

$$ \text{E-Swish}_{\gamma}(x) = γ  x ⋅ \text{sigmoid}(x)$$


### Mish Activation

Mish is similar to Swish, smooth, continuous, non-monotonic, unbounded above and bounded below. It can be defined as:

$$ \text{Mish}(x) = x \text{tanh}(\text{softplus}(x))$$

It is effectively solving dead neuron and vanishing gadient problem, and usually outperform Swish, and ReLU.

### Gaussian Error Linear Units (GELU)

__GELU__  is one of the most promising activation function. It is widely used in BERT, GPT-3 and other transformers. GELU weight inputs by their values, rather than gated sign as in ReLU. GELU scales input $x$ by how much greater it is than other inputs. It can be expressed as: 

$$\text{GELU}(x) = x ⋅ Φ (x) \notag$$

Where, $\Phi(x)$ is cumulative distribution function of standard Gaussian distribution. Thus, GELU can be defined as: 

$$\text{GELU}(x) = x ⋅ \frac{1}{2}\left[1 + \text{erf}(x / \sqrt{2}) \right]$$

GELU can be approximated with $0.5\,x(1 + \text{tanh}[\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)])$ (Hendrycks & Gimpel, 2016)


<!-- ### Softmax

Softmax activation function is used in the final layer of network for multi-class classification tasks. It maps output as a probability distribution in the range of $[0,1]$ and sum of each outcome is equal to $1$. Given a input vector $\overrightarrow z $ with $K$ classes, softmax can be defined as:

$$\text{softmax}(\overrightarrow z) = \frac{e^{z_{i}}}{\sum_{j =1}^K e^{z_{i}}} $$ -->

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
- Hendrycks, D. and Gimpel, K., 2016. Gaussian error linear units (gelus). _arXiv preprint arXiv:1606.08415_.
- Ramachandran, P., Zoph, B. and Le, Q.V., 2017. Searching for activation functions. _arXiv preprint arXiv:1710.05941_.
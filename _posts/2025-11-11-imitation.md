---
title: Imitation Learning
tags: imitation-learning reinforcement-learning
sidebar:
  nav: "docs-en"
mathjax: true
mathjax_autoNumber: true
---

## __Imitation Learning__

Consider a standard Reinforcement Learning case of finite horizon episodic Markov Decisin Process (MDP) with horizon $\mathcal{H}$, state space, $\mathcal{S}$ and action space $\mathcal{A}$

Then, in __Imitation Learning__, instead of interacting directly with the environment to learn from, we are given a dataset

$$\mathcal{D} = \{(s_{i}^h, a_{i}^h,)\}_{i=1\cdots N, h = 1 \cdots H}$$

consisting of $N^H$ independent trajectories (each of length $\mathcal{H}$) sampled from a fixed unknown expert policy $\pi^E$

__Objective of Imitation Learning:__ To Learn a policy close to optimal, assuming expert policy is close to optimal such that:

$$V_{1}^*(s_1) - V_{1}^{\pi^E} (s_1) \le \epsilon_E$$ for some small $ϵ_E \gt 0$

## __Behaviour Cloning (BC)__

It is the simplest method of imitation learning and directly learning to replicate the expert policy in a supervised manner. Precisely, the __behavior cloning policy__, $\pi^{BC}$ is obtained as:

$$\pi^{BC} \in \text{arg min}_{\pi \in \mathcal{F}} \sum_{h=1}^H \left(\mathbb{E}_{\left(s_{i}^h,a_{i}^h\right)} \left[- \log \pi_{h}\left(a_{i}^h | s_{i}^h\right)\right] + R_{h}(\pi_{h})\right)$$

where,

- $\mathcal{F} = \{\pi \in \Pi: \pi_{h} \in \mathcal{F}_h\}$ is the policy class (product over horizon h)
- Each $\pi_h : \mathcal{S} → Δ(\mathcal{A})$ is a conditional dsitribution mapping states to action distributions.
- $R_{h}$ is some regularizer (e.g KL-Divergence penalty, weight decay, etc)

In __BC__ afterwards, in order to provider convergence guarantee, we make two assumptions

- __Assumption 1:__ We assume, some regularity conditions on the class of policies defined in terms of _covering numbers of the class_. This ensures class gets to generalize well, by bounding the number of distinct policies needed to approximate the entire class.

For all $h \in [\mathcal{H}]$, and $ϵ \in (0,1)$ there are two positive constants, $d_{\mathcal{F}}, \mathbb{R}_{\mathcal{F}} \gt 0$ such that :

$$\log \mathcal{N} (\epsilon, \mathcal{F}_{\mathcal{h}}, \lVert \cdot \rVert_{∞}) \le d_{\mathcal{F}}\log(R_{\mathcal{R}/ϵ})$$

Moreover, there is a constant $\gamma \gt 0$ such that, it holds $\pi_{h}(a \mid s) \ge \gamma$ for any $(s, a) \in \mathcal{S} \times \mathcal{A}$. This ensures for every policy in class $F$, probability of taking any action will never be zero, this prevents negative log-likelihood, $-\log(\pi_{h}(a \mid s))$ to turn $-\log(0)$

- __Assumption 2:__ We assume that a smooth version of the expert policy belongs to the class hypothesis space, $\mathcal{F}$:

$$\pi_{h}^{E, κ} (a|s) = (1-\kappa)\pi_{h}^E (a|s) + \frac{\kappa}{\lvert A \rvert}$$

where, $\kappa \in (0, 1/2)$ is a small exploration noise, ensuring all actions have non-zero support, so KL-Divergence remain finite.

## __DAgger__

__Dataset Aggregation (DAgger)__ algorithm start by using the expert's policy at first iteration to gather a dataset of trajectories $\mathcal{D}$ and trains a policy $\hat{\pi_{2}}$ that best mimics the expert on those trajetories.

Afterwards, at next iterations of $n$, it uses $\hat{\pi_{n}}$ to collect more trajectories and adds those trajetories to the dataset $\mathcal{D}$. The next policy $\hat{\pi_{n+1}}$ is the policy that best mimics the expert on the whole dataset $\mathcal{D}$

More clearly, we leverage the presence of expert in the first iteration and slightly reduce over the $n$ iteration with

$$\pi_{i} = \beta_{i}\pi^* + (1-\beta_{i})\hat{\pi_{i}}$$

We use, $\beta_1 = 1$ for the first $\hat{\pi_{1}}$, and decay it exponentially with $\beta_{i} = p^{i-1}$

__Preliminaries:__

If the learner executed policy $\pi$ from time step 1 to $t-1$, then $d_{\pi}^t$ denote the distribution of states at time $t$ for the policy $\pi$, and we can express the average distribution of states as:

$$d_{\pi} = \frac{1}{T}\sum_{t=1}^T d_{\pi}^t$$ if we follow $\pi$ for $T$ steps.

Next, we use $C(s,a)$ as the expected cost of taking action $a$ in state $s$ following $\pi$, which can be expressed as:

$$C_{\pi}(s) = \mathbb{E} a \sim \pi(s) [C(s,a)]$$

Here, we assume that $C$ is bounded between in $[0,1]$.Thus, the total cost of executing policy $\pi$ for $T-$steps can be denoted as:

$$J(\pi) = \sum_{t=1}^T \mathbb{E}_s \sim d_{\pi}^t [C_{\pi}(s)] = T\mathbb{E}_s \sim d_{\pi} [C_{\pi}(s)]$$

In __imitation learning__, we do not get to see the true cost, $C(s,a)$, therefore we seek to bound $J(\pi)$ for any cost function $C$ based on how well $\pi$ mimics the expert's policy $\pi^*$.

Let $l$ be a surrogate loss function that we minimize instead of $C$, as we don't know $C$.

Our goal is to find a policy $\hat{\pi}$ which minimizes the observed surrogate loss under its induced distribution of states i.e.,

$$\hat{\pi} = \text{arg min}_{\pi \in \Pi} \mathbb{E}_{s \sim d_{\pi}}[l(s,\pi)]$$

__What DAgger actually does:__

Once, we deploy the learner, following $\pi$, it start to induce state distribution $d_{\pi}$. If the learner policy make even smaller mistake, state induced could be absolutely random, leading to more random state distribution in collection.

Therefore, only minimizing $
l(s, \pi)$ doesn't guarantee low $J(\pi)$.

This is why, we have discussed above,

$$\pi_{i} = \beta_{i}\pi^* + (1-\beta_{i})\hat{\pi_{i}}$$

We use, $\beta_1 = 1$ for the first $\hat{\pi_{1}}$, where it learns completely from expert in first iteration gradually phase out.

Next, __data is collected__ $(s, \pi^*(s))$ from the states actually encountered under $\pi_{i}$ and its distribution $d_{\pi_{i}}$

Afterwards ,we aggregate the dataset:

$$\mathcal{D} \leftarrow D  \cup {(s, \pi^* (s))}_{s \sim d_{\pi_{i}}}$$

and, train the next policy $\hat{\pi_{i+1}}$ to minimize empirical loss on the aggregate dataset

$$\hat{\pi_{i+1}} ≃ \text{arg min} \frac{1}{i} \sum_{j=1}^i \mathbb{E}_{s\sim d_{\pi_{j}}} [l(s,\pi)]$$

Evetually, after $N$ iterations, DAgger has aggregated data from distributions ${d_{\pi_{1}}}\cdots d_{\pi_{n}}$ as:

$d_{N} =\frac{1}{N}\sum_{i=1}^N d_{\pi_{i}}$.

If we take $d_{N}$ as an approximiation for $d_{\pi}$, the data distribution final learnt policy will induces, then we policy as:

$$\hat{\pi} = \text{arg min}_{\pi \in\Pi} \mathbb{E}_s\sim d_{\hat{\pi}} [l(s,\pi)]$$

Thus, reaching goal!

## __GAIL__

__Generative Adversarial Imitation Learning (GAIL)__ uses maximum causal entropy of Inverse Reinforcement Learning (IRL) and Generative Adversarial networks to build its optimization solution.

We know that, maximum causal entropy of IRL which fits a cost function from a family of functions $C$ with the optimization problem is as:

$$\text{maximize}_{c\in C}\left(\min\limits_{\pi \in \Pi} - \mathcal{H}(\pi) + \mathbb{E}[c(s,a)]\right) - \mathbb{E}_{\pi_{E}}[c(s,a)]$$

Where, $\mathcal{H}(\pi) ≜ \mathbb{E}_{\pi}[-log \,\pi\,(a \mid s)\,]$ is the $\gamma-$discounted causal entropy of the policy $\pi$

__Occupancy Measure__ : It is the distribution of state-action pairs that an agent would encounter when navigating the environment with policy $\pi$, and it allow us to write $\mathbb{E}_{\pi}[c(s,a)] = \sum_{s,a}\rho_{\pi}(s,a)c(s,a)$ for any cost function $c$.

More specifically, __Occupancy Measure__
$\rho_{\pi}: \mathcal{S}\times \mathcal{A} \rightarrow \mathbb{R}$ can be defined as:

 $$\rho_{\pi}(s,a) = \pi(a \mid s) \sum_{t=0}^{\infty} \gamma^t P(s_t = s \mid \pi)$$

 Next, as in GAN, where we have a discriminator $D$ and a generator $G$. The job of $D$ is to distinguish between the data distribution generated by $G$ and the true data distribution.

In our setting, the __learner's occupancy measure__ $\rho_{\pi}$ is analogous to the data distribution generated by $G$ and the __expert's occupancy measure $\rho_{\pi_{E}}$__ is analogous to the true data distribution.

Thus, our imitation learning algorithm objective can be written as:

$$\min\limits_{\pi}  \psi_{GA}^*(\rho_{\pi} - \rho_{\pi_{E}}) - \lambda \mathcal{H}(\pi)$$

Here, $\psi_{GA}^*$ is cost regularizer that tries to distinguish samples from occupancy measures $\rho_{\pi}$ and $\rho_{\pi_{E}}$ and further stated as:

$$\psi_{GA}^*(\rho_{\pi} - \rho_{\pi_{E}}) = \text{max}_{D \in (0,1)^{\mathcal{S}\times \mathcal{A}}} \mathbb{E}_{\pi}[\log (D(s,a))] + \mathbb{E}_{\pi_{E}}[1 -\log D(s,a)]$$

Thus, __formally GAIL method can be expressed as__:

$$\min\limits_{\pi}\max\limits_{D} \; \mathbb{E}_{\pi}[\log (D(s,a))] + \mathbb{E}_{\pi_{E}}[1 -\log D(s,a)] - \lambda \mathcal{H}(\pi)$$

In practice,

- We parameterize policy $\pi$ by $\pi_{\theta}$ (with network weight $\theta$) and,
- parameterize discriminator $D$ by $D_{w}$ (with weight $w$)
- While sampling trajectories $\tau_{i} \sim \pi_{\theta_{i}}$ from the dataset of expert trajectories $\tau_{E} \sim \pi_{E}$
- then, we update gradient step on $w$ to increase with respect to $D$ as in:

$$\hat{\mathbb{E}}_{\tau \sim \pi_{\theta}} [\log D_w (s,a)] + \hat{\mathbb{E}}_{\tau_{E}} [\log(1-D_w (s,a)]$$

- and use TRPO step on $\theta$ to decrease with respect to $\pi$, specifically take a KL-constrained natural gradient step with:

$$\hat{\mathbb{E}}_{\tau_{i}}[\nabla_{\theta} \log \pi_{\theta}(a \mid s) Q(s,a)] - \lambda \nabla_{\theta}\mathcal{H}(\pi_{\theta})$$

where,

$$
Q(\bar{s}, \bar{a}) = \hat{\mathbb{E}}_{\tau_i} \left[ \log\big(D_{w_{i+1}}(s, a)\big) \,\middle|\, s_0 = \bar{s},\, a_0 = \bar{a} \right]
$$

## __SQIL__

__Soft Q Imitation Learning__ brings three modification with soft Q-Learning.

- It initially fills the agent's experience replay buffer with demonstrations, with __reward = +1__. It is represented here as $D_{demo}$
- When agent interacts and accumulates new experiences, it adds them in replay buffer with __reward = 0__. It is represented as $D_{samp}$

- It further balances the number of demonstration experience and new experience at __50% each__ in each sample from the replay buffer. Half a minibatch from both $\mathcal{D}_{demo}$ and $\mathcal{D}_{samp}$

Since, soft Q-learning is an off-policy algorithm, the agent does not need to visit demonstrated states to get positive reward, it can replay demonstration that were initially added to its buffer. Thus, SQIL can be used in stochastic environments where demonstrated states may never be visited again.

SQIL algorithm can be expressed as below:

$$
\delta^2(\mathcal{D}, r) \triangleq \frac{1}{\lvert \mathcal{D} \rvert} \sum_{s, a, s' \in \mathcal{D}} 
\left( Q_{\theta}(s, a) - \left( r + \gamma \log \left( \sum_{a' \in \mathcal{A}} \exp\big( Q_{\theta}(s', a') \big) \right) \right)\right)^2 $$

where, $r \in \{0,1\}$, $Q_{\theta}$ is the soft Q function, $\mathcal{D}_{demo}$ are demonstrations and $\delta^2$ is the __squared soft Bellman error__.

The inner bracket is the soft Bellman error which we are squaring to further penalize over and under-estimation, and then take average over all transition in dataset $D$.

While running, objective function of SQIL is to minimize weighted sum of bellman error for demo and sampled data and update gradient step as following:

$$\theta \leftarrow \theta - \eta \nabla_{\theta} \left( \delta^2 (\mathcal{D}_{demo},1) + λ_{samp} \delta^2(\mathcal{D},0)\right)$$

Over time, $\mathcal{D}_{samp}$ will start to grow and dominate the replay buffer. If we naively sample uniformly, almost all samples will have $r=0$ and the positive reward will effectively decay to zero.

This is helped by third modification where we sample $50\%$ from each 
$\mathcal{D}_{samp}$ and $\mathcal{D}_{demo}$ with effective reward as $r_{eff} = \frac{1}{1 + \lambda_{samp}}$.

## __AIRL__

In Reinforcement Learning, we aim to find optimial policy $\pi^*$ that maximises the entropy-regularized discounted reward function $r(s,a)$ when we have dynamic or transition distribution $\mathcal{T}(s' \mid a,s)$ and the initial state distribution $\rho_{0}(s)$ are known and can only be queried through MDP. It is expressed as:

$$\pi^* = \text{arg max}_{\pi} E_{\tau \sim \pi} \left[\sum_{t=0}^T \gamma^t(r(s_t, a_t) + \mathcal{H}(\pi(\cdot \mid s_t))) \right]$$

where, $\tau = \{s_0, a_0, \cdots s_t, a_t\}$

__Inverse Reinforcement Learning__ flips this problem. It seeks to infer, instead, the reward function $r(s,a)$ given a set of demonstration $\mathcal{D} = \{\tau_1, \tau_2,
\cdots \tau_N\}$ assuming these demonstrations are drawn from an optimal policy $\pi^*(a|s)$. It can be interpreted as solving the maximum likelihood problem:

$$\max_{\theta} E_{\tau \sim \mathcal{D}} [\log p_{\theta}(\tau)]$$

where $p_{\theta}(\tau) ∝ p(s_{0}) \Pi_{t=0}^T p(s_{t+1} \mid s_t, a_t)\mathcal{e}^{\gamma^t r_{\theta}(s_t, a_t)}$

It parameterizes the reward function, and calculating it requires knowing partition function i.e, the sum over al the trajectories $\tau_n$ which is intractable.

Therefore, __Generative Adversaial Network Guided Cost Learning (GAN-GCL)__ is proposed where we have a $(a)$ discriminator of form $f_{\theta}(\tau)$ and updating it is viewed as updating the reward function, and $(b)$ a policy $\pi(\tau)$ which produces trajectories and updating it can be viewed as improving the sampling distribution to estimate partition function. It can be expressed as:

$$D_{\theta}(\tau) = \frac{\text{exp}\{f_{\theta}(\tau)\}}{\text{exp}\{ f_{\theta}(\tau) + \pi(\tau)\}}$$

with policy objective to maximise $R(\tau) = \log(1 - D(\tau)) - \log D(\tau)$

If further trained to optimality, the learned function would converge to __optimal reward function__ $f^* (\tau) = R^* (\tau) + \text{const}$ and policy $\pi$ converges to __optimal policy__ $\pi^*$. This is what makes GAN-GCL different from GAIL where reward cannot be recovered.

__Adversarial Inverse Reinforcement Learning (AIRL):__

Using full trajectories as proposed by GAN-GCL result in high variance estimate and poor learning. Therefore, AIRL solves this discriminating not on trajectory-level but state-action level. This reduces variances and stablises training for every time-step for (s,a) pair. It can be written as:

$$D_{\theta}(s,a) = \frac{\text{exp}\{f_{\theta}(s,a)\}}{\text{exp}\{f_{\theta}(s,a)\} + \pi(a \mid s)}$$

In this new discriminator, at optimality it does not converge to reward function, but __Advantage Function__ $$f^*(s,a) = \log(\pi)^*(a \mid s)  = A^*(s,a)$$

Advantange function, makes AIRL an efficient algorithm for imitation learning, but it is not a valid optimal reward function but a heavily _entagled_ reward with the old environment dynamics.

To decouple true reward function $r_{\theta}(s)$ from the advantage function, discriminator is further modified as:

$$D_{\theta}(s,a,s') = \frac{\text{exp}\{f_{\theta},\phi (s,a,s') \}}{\text{exp}\{f_{\theta},\phi (s,a,s') \} + \pi(a|s)}$$

where, $f_{\theta},\phi$ is restricted to a reward approximator $g_{\theta}$ and a shaping term $h_{\phi}$ as:

$$f_{\theta},\phi (s,a,s')= g_{\theta}(s,a) + \gamma h_{\phi}(s') - h(\phi)(s)$$

$f_{\theta},\phi (s,a,s')$ is still our advantage function that now combines __shaping term__ to mitigate the effects of unwanted shaping, and __reward approximator__ $g_{\theta}(s)$ which is a function of state now, allowing us to extract reward that are distangled from environment dynamics.

Under deterministic environment with a state-only ground truth reward, we can get these as:

$$g^*(s) = r^*(s) + \text{const}$$

$$h^*(s) = V^*(s) + \text{const}$$

where $ r^* $ is the true reward function. $ h^* $ recovers the optimal value function $ V^* $, which serves as the reward shaping term.

## Reference

- Fu, J., Luo, K. and Levine, S., 2017. Learning robust rewards with adversarial inverse reinforcement learning. arXiv preprint arXiv:1710.11248
- Ho, J. and Ermon, S., 2016. Generative adversarial imitation learning. Advances in neural information processing systems, 29
- Reddy, S., Dragan, A.D. and Levine, S., 2019. Sqil: Imitation learning via reinforcement learning with sparse rewards. arXiv preprint arXiv:1905.11108
- Ross, S., Gordon, G. and Bagnell, D., 2011, June. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics (pp. 627-635). JMLR Workshop and Conference Proceedings.
- Tiapkin, D., Belomestny, D., Calandriello, D., Moulines, E., Naumov, A., Perrault, P., Valko, M. and Menard, P., 2023. Regularized rl. arXiv preprint arXiv:2310.17303
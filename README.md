# E²M: Double-Bound α-Divergence Optimization for Tensor-Based Mixture Discrete Density Estimation

[![arXiv](https://img.shields.io/badge/arXiv-2405.18220-b31b1b.svg)](https://arxiv.org/abs/2405.18220)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

This repository provides a non-negative tensor decomposition algorithm for optimizing the α-divergence using a double-bound strategy. The method has the following key advantages:

1. ✨**No More Learning Rate Tuning**  
   Our approach is based on a double-bound EM algorithm, where all parameters are updated simultaneously at each step. It does not rely on gradient-based optimization, eliminating the need for learning rate tuning.

3. ✨**Unified Framework for Tensor Decomposition**  
   The algorithm supports a wide range of low-rank tensor structures, including CP, Tucker, Tensor Train, their mixtures, and adaptive background terms. It handles both sparse and dense tensors with various objective functions, including KL divergence (α = 1.0) and Hellinger distance (α = 0.5).

5. ✨**Robustness Control via α-Divergence**  
   Optimizing the α-divergence enables control over sensitivity to outliers and noise. When α → 0 (reverse KL divergence), the algorithm exhibits mass-covering behavior that ignores small outliers; when α = 1 (KL divergence), it shows mode-seeking behavior that focuses on dense regions. This implementation supports 0 < α ≤ 1.

6. ✨**Convergence Guarantee**  
   The objective function is guaranteed to decrease monotonically at each iteration, ensuring convergence.

## Environment

- Python 3.12.3  
- NumPy 2.2.2

## Authors and Citation

This is joint work by [Kazu Ghalamkari](https://gkazu.info/), [Jesper Løve Hinrich](https://www2.compute.dtu.dk/~jehi/), and [Morten Mørup](https://mortenmorup.dk/) at [DTU Compute](https://www.compute.dtu.dk/).  

For technical details and theoretical background, see the preprint:  
[E2M: Double-Bound α-Divergence Optimization for Tensor-Based Discrete Density Estimation](https://arxiv.org/abs/2405.18220)


## How to Run

We provided two demo files as a Jupyter notebook.
1. [demo_dense.ipynb](https://github.com/gkazunii/eemix/blob/main/demo/demo_dense.ipynb)
2. [demo_sparse.ipynb](https://github.com/gkazunii/eemix/blob/main/demo/demo_sparse.ipynb)

Just click and open the demo files. All necessary instructions are described there.

## FQA

#### What is the novelty of this algorithm?

EM-based tensor factorization methods for CP decomposition that optimize the Kullback-Leibler (KL) divergence have been previously studied in [1], [2]. This repository generalizes their approaches to support not only various low-rank tensor structures such as CP, Tucker, Tensor Train, their mixtures, and background terms, but also optimization under the α-divergence. 

In general, optimizing α-divergence is challenging due to the power term involved, which prevents closed-form updates of all parameters. Our method overcomes this difficulty using a double-bound strategy, described below, enabling closed-form updates for all steps for many low-rank structures.


#### What is the relationship between tensors and probability distributions?

A normalized nonnegative tensor can be interpreted as a discrete probability distribution. Under this interpretation, low-rank tensor decomposition corresponds to identifying latent variables, and the EM algorithm can be used to optimize the KL divergence between the observed and reconstructed distributions.

This probabilistic view allows tensor factorization to be seen as a latent variable model, making it possible to apply information-theoretic techniques.

#### What theory is behind the algorithm?

We show that the α-divergence can be bounded by a sequence of KL divergences via Jensen’s inequality. Specifically:

    Applying Jensen’s inequality to α-divergence yields a bound in terms of KL divergence.

    Applying Jensen’s inequality to KL divergence gives rise to the Evidence Lower Bound (ELBO), as in standard EM.

Our E²M algorithm optimizes α-divergence by iterative three steps:

    M-step: Maximizes the ELBO with respect to model parameters.

    E1-step: Tightens the bound from ELBO to KL divergence.

    E2-step: Tightens the bound from KL divergence to α-divergence.

Both E1 and E2 steps admit closed-form updates. The M-step benefits from a key property: since the ELBO contains no sum inside the logarithm, many low-rank structures decouple into independent subproblems. This enables simultaneous, closed-form updates of all parameters. Moreover, the M-step is equivalent to a many-body approximation, which becomes a convex optimization regardless of the low-rank structure. 

In short, the proposed algorithm can be viewed as an EM framework for tensor multi-body decomposition with latent variables, extending information geometric interpretations (see the case α → 1 discussed in [3]).

#### What is α-divergence, and why optimize it?

The α-divergence from a given tensor T to a reconstructed low-rank tensor P is defined as:

$$
D_{α}(T,P)=\frac{1}{α(1-α)}\sum_{i_1,\dots,i_D} ( 1 - T_{i_1,\dots,i_D}^α P_{i_1,\dots,i_D}^{1-α} ) 
$$

This divergence family includes:

    the KL divergence as α → 1,

    the Hellinger distance as α → 0.5

    the reverse KL divergence as α → 0.

In our algorithm, α is treated as a hyperparameter. As illustrated in the attached figure, α controls the sensitivity of the reconstruction to outliers or noise. 

#### What are the advantages compared to gradient-based methods?

Gradient-based methods require careful tuning of learning rates. Our E²M algorithm achieves similar or better optimization performance without learning rate tuning. Empirically, we demonstrate that the closed-form updates of our method match or exceed the efficiency and stability of well-tuned gradient-based approaches.

## License

This source code is released under the MIT License.

# E²M: Double-Bound α-Divergence Optimization for Tensor-Based Mixture Discrete Density Estimation

[![arXiv](https://img.shields.io/badge/arXiv-2405.18220-b31b1b.svg)](https://arxiv.org/abs/2405.18220)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

This repository provides a non-negative tensor decomposition algorithm for optimizing the α-divergence using a double-bound strategy. The method has the following advantages:

1. ✨**No More Learning Rate Tuning**  
   Our approach is based on a double-bound EM algorithm, where all parameters are updated simultaneously at each step. It does not rely on gradient-based optimization, eliminating the need for learning rate tuning.

3. ✨**Unified Framework for Tensor Decomposition**  
   The algorithm supports a wide range of low-rank tensor structures, including CP, Tucker, Tensor Train, their mixtures, and adaptive background terms. It handles both sparse and dense tensors with various objective functions, including KL divergence (α = 1.0) and Hellinger distance (α = 0.5).

5. ✨**Robustness Control via α-Divergence**  
   Optimizing the α-divergence enables control over sensitivity to outliers and noise. When α → 0 (reverse KL divergence), the algorithm exhibits mass-covering behavior that ignores small outliers; when α = 1 (KL divergence), it shows mode-seeking behavior. This implementation supports 0 < α ≤ 1.

6. ✨**Convergence Guarantee**  
   The objective function is guaranteed to decrease monotonically at each iteration, ensuring convergence.

## 🛠️ Environment

- Python 3.12.3  
- NumPy 2.2.2

## 🧑‍💻 Authors and Citation

This is joint work by  

- [Kazu Ghalamkari](https://gkazu.info/) <a href="https://orcid.org/0000-0002-4779-2856"><img src="https://upload.wikimedia.org/wikipedia/commons/0/06/ORCID_iD.svg" width="16" /></a>,  
- [Jesper Løve Hinrich](https://www2.compute.dtu.dk/~jehi/) <a href="https://orcid.org/0000-0003-0258-7151"><img src="https://upload.wikimedia.org/wikipedia/commons/0/06/ORCID_iD.svg" width="16" /></a>,  
- [Morten Mørup](https://mortenmorup.dk/) <a href="https://orcid.org/0000-0003-4985-4368"><img src="https://upload.wikimedia.org/wikipedia/commons/0/06/ORCID_iD.svg" width="16" /></a>,

at 🎓 [DTU Compute](https://www.compute.dtu.dk/).

For technical details and theoretical background, see the preprint:  
📄 [E²M: Double-Bound α-Divergence Optimization for Tensor-Based Discrete Density Estimation](https://arxiv.org/abs/2405.18220)


## 🚀 How to Run

We provided two demo files as a Jupyter notebook.
📘 [demo_dense.ipynb](https://github.com/gkazunii/eemix/blob/main/demo/demo_dense.ipynb)
📘 [demo_sparse.ipynb](https://github.com/gkazunii/eemix/blob/main/demo/demo_sparse.ipynb)

Just click and open the demo files. All necessary instructions are described there.

## ❓FAQ

#### 💡What is the novelty of this algorithm?

EM-based tensor factorization methods for CP decomposition that optimize the Kullback-Leibler (KL) divergence have been previously studied in [[1](https://ieeexplore.ieee.org/abstract/document/8335432)] and [[2](https://ieeexplore.ieee.org/document/8821380)]. This repository generalizes their approaches to support not only various low-rank tensor structures such as CP, Tucker, Tensor Train, their mixtures, and adaptive background terms, but also optimization under the α-divergence. In general, α-divergence optimization is challenging due to its power term, which prevents closed-form updates of all parameters. Our method overcomes this difficulty using a double-bound strategy, described below.


#### 💡What is the relationship between tensors and probability distributions? Why can we apply the EM-based method for tensor decomposition?

A normalized nonnegative tensor can be interpreted as a discrete probability distribution. Specifically, each tensor element $T_{i_1,\dots,i_D}$ can be regarded as $p(x_1=i_1,x_2=i_2,\dots,x_D=i_D)$. The index set $[I_1]\times[I_2]\times\dots\times[I_D]$ is regarded as the sample space $\Omega$. When we assume the CP-low-rank structure in the tensor $T$, it can be written as $T_{i_1,\dots,i_D}=\sum_{r} Q_{i_1,\dots,i_D,r}$ where the normalized non-negative higher-order tensor $Q$ is given as $Q_{i_1,\dots,i_D,r}=A^1_{i_1,r} \dots A^D_{i_D,r}$. The model $P$ is summed up over the $r$, and if we regard $r$ as the hidden variable and $\sum_r$ as a marginalization, we can adapt the EM-algorithm, which is a well-known approach for maximum likelihood estimation with the model with hidden variables. 


#### 💡What theory is behind the algorithm?

The α-divergence can be bounded by the KL divergences using Jensen’s inequality, and KL divergences can be bounded by the ELBO using Jensen’s inequality again. Our E²M algorithm optimizes the α-divergence by iterative three steps:

    M-step: Maximizes the ELBO with respect to model parameters.

    E1-step: Tightens the bound from ELBO to KL divergence.

    E2-step: Tightens the bound from KL divergence to α-divergence.

Both E1 and E2 steps admit closed-form updates. The M-step benefits from a key property: since the ELBO contains no sum inside the logarithm, many low-rank structures decouple into independent subproblems. This enables simultaneous closed-form updates of all parameters. Moreover, the M-step is equivalent to a many-body approximation, which becomes a convex optimization regardless of the low-rank structure, even if the closed-form updates are not available. In short, the algorithm can be viewed as an EM framework for tensor many-body approximation with hidden variables [[3](https://openreview.net/forum?id=5yedZXV7wt)]. 

#### 💡What is α-divergence, and why optimize it?

The α-divergence from a given tensor T to a reconstructed low-rank tensor P is defined as:

$$
D_{α}(T,P)=\frac{1}{α(1-α)}\sum_{i_1,\dots,i_D} ( 1 - T_{i_1,\dots,i_D}^α P_{i_1,\dots,i_D}^{1-α} ) 
$$

This divergence family includes:

    the KL divergence as α → 1,

    the Hellinger distance as α = 0.5,

    the reverse KL divergence as α → 0.

In our algorithm, α is treated as a hyperparameter. α controls the sensitivity of the reconstruction to outliers and noise. Please refer to the [demo file](https://github.com/gkazunii/eemix/blob/main/demo/demo_dense.ipynb) to confirm the robustness of the outliers.

#### 💡What are the advantages compared to gradient-based methods?

Gradient-based methods require careful tuning of learning rates. Our E²M algorithm achieves similar or better optimization performance without learning rate tuning.


#### 💡What is the computational complexity per iteration?

Since the closed-update formula in the M-step for the CP, Tucker, and Tensor Train structures is proportional to the input $I\times\dots\times I$ tensor $T$, the computational complexity is proportional to the number of nonzero elements in $T$. Specifically, the computational complexity per iteration is $O(DNR)$ for the CP structure, $O(DNR^D)$ for Tucker structure, and $O(NDR^2)$ for the Tensor Train structure, where $N$ is the number of nonzero element in $T$, $R$ is the tensor rank, and $D$ is the order of the tensor $T$. We emphasize that computational complexity is not proportional to $I^D$.

#### 💡Can we apply the algorithm to a non-normalized tensor?

Our method assumes that the input tensor is normalized. If the tensor is not normalized, the following heuristic can be applied. First, record the total sum $\lambda$ of the input tensor $T$. Then, normalize the input tensor and apply the E²M algorithm. Finally, multiply the reconstructed tensor by $\lambda$. Please refer to the example for image reconstruction in the [demo file](https://github.com/gkazunii/eemix/blob/main/demo/demo_dense.ipynb). In the case of optimizing the KL divergence for positive measures (often called I-divergence), the sum of the reconstructed tensor is the same as the sum of the input tensor, so this heuristic is reasonable.

#### 💡Can we apply the algorithm to a real-valued tensor?

No. Non-negativity is an essential assumption of the E²M algorithm.

## 📜 License

This source code is released under the MIT License.
Feel free to use, remix, and share responsibly! 🔓✨

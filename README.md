# E2M: Double-Bound α-Divergence Optimization for Tensor-Based Mixture Discrete Density Estimation

[![arXiv](https://img.shields.io/badge/arXiv-2405.18220-b31b1b.svg)](https://arxiv.org/abs/2405.18220)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

This repository provides a tensor decomposition algorithm for optimizing the α-divergence using a double-bound strategy. The method has the following key advantages:

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

## License

This source code is released under the MIT License.

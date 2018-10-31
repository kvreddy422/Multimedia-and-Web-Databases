# *TensorD*: A Tensor Decomposition Library in TensorFlow

[![Build Status](https://travis-ci.org/Large-Scale-Tensor-Decomposition/tensorD.svg?branch=master)](https://travis-ci.org/Large-Scale-Tensor-Decomposition/tensorD)

Tensor:D

## What is *TensorD*?

*TensorD* is a Python tensor library built on ``TensorFlow``  [1]. It provides basic decomposition methods, such as Tucker decomposition and CANDECOMP/PARAFAC (CP) decomposition, as well as new decomposition methods developed recently, for example, Pairwise Interaction Tensor Decomposition. 



*TensorD* is designed to be flexible, lightweight and scalable when used to transform idea into result as soon as possible in nowadays research. Based on ``TensorFlow``, *TensorD* has several key features:

- **GPU Compatibility**: *TensorD* is completely built within ``TensorFlow``, which enables all GPUs to be visible to the process [2] and flexible usage on GPU computation for acceleration.
- **Static Computation Graph**: *TensorD* runs in Static Computaiton Graph way, which means defining computation graph at first then running real computaion with dataflow. 
- **Light-weighted**: *TensorD* is written in Python which provides high-level implementations of mathematical operations. Acquiring small memory footprint, *TensorD* is friendly to install even on mobile devices.
- **High modularity of structure for extensibility**: *TensorD* has a modular structure which facilitates the expansion optionally. *TensorD* modulizes its code for the convenience of using its tensor classes, loss functions, basic operations and decomposition models separately as well as plugged together. 
- **High-level APIs**: The tensor decomposition part in *TensorD* is object-oriented and high-level interface on TensorFlow, which facilitates direct using. The purpose of such design is that users can make simple calls without knowing the details of implementations.
- **Open Source and MIT Licensed**: *TensorD* uses MIT license, and is an open source library in Tensorflow. Everyone can use and modify according to their own specific applications.





## Structure

![Structure of TensorD](https://github.com/Large-Scale-Tensor-Decomposition/tensorD/raw/master/pictures/struct.png)

*TensorD*'s implementations of structure are clear and modular. The library structure is roughly contains three main modules: 

1) Data processing module, providing interface to read and write sparse tensor in coordinate format, and a transformation between sparse and dense tensor.
2) Basic operation module, which assembled via the linear algebra in TensorFlow, providing basic matrix and tensor operations not only for tensor decomposition but also for other algorithms.
3) Decomposition model module, including common decomposition algorithms such as CP decomposition [3, 4, 5], Tucker decomposition [6, 7], NCP decomposition [8, 9] and NTucker decomposition [8, 10]





## Requirements

- Python 3.x
- TensorFlow(>=1.2.1) (see  [Installing TensorFlow](https://www.tensorflow.org/install/)),
- NumPy

## Installation

Clone the *TensorD* repository:

```
git clone https://github.com/Large-Scale-Tensor-Decomposition/tensorD.git
```



##  Example

Here is a simple example for CP decomposition. See [documentation](http://smile-lab-tensord.readthedocs.io) or [demo files](https://github.com/Large-Scale-Tensor-Decomposition/tensorD/tree/master/tensorD/demo) for more examples. 

```python
# import necessary packages
from tensorD.factorization.env import Environment
from tensorD.dataproc.provider import Provider
from tensorD.factorization.cp import CP_ALS
import tensorD.demo.DataGenerator as dg

# use synthetic_data_cp to generate a random tensor with shape of 40x40x40
X = dg.synthetic_data_cp([40, 40, 40], 10)
data_provider = Provider()
data_provider.full_tensor = lambda: X
env = Environment(data_provider, summary_path='/tmp/cp_demo_' + '30')
cp = CP_ALS(env)
# set rank=10 for decomposition
args = CP_ALS.CP_Args(rank=10, validation_internal=1)
# build decomposition model with arguments
cp.build_model(args)
# train decomposition model, set the max iteration as 100
cp.train(100)
# obtain factor matrices from trained model
factor_matrices = cp.factors
for matrix in factor_matrices:
    print(matrix)
# obtain scaling vector from trained model
lambdas = cp.lambdas
print(lambdas)
```



## License

*TensorD* is released under the MIT License (refer to LISENSE file for details).



## Reference

[1] M. Abadi, P. Barham, J. Chen, Z. Chen, A. Davis, J. Dean, M. Devin, S. Ghemawat,G. Irving, M. Isard, et al., Tensorflow:  A system for large-scale machine learning., in:  OSDI, Vol. 16, 2016, pp. 265-283.

[2] Using gpus, https://www.tensorflow.org/tutorials/using_gpu .

[3] H. A. Kiers, Towards a standardized notation and terminologyin multiway analysis, Journal of chemometrics 14 (3) (2000)105–122.

[4] J. Mocks, Topographic components model for event-related potentials and some biophysical considerations, IEEE transactions on biomedical engineering 35 (6) (1988) 482–484.

[5] J. D. Carroll, J.-J. Chang, Analysis of individual differences inmultidimensional scaling via an n-way generalization of eckart-young decomposition, Psychometrika 35 (3) (1970) 283–319.

[6] F. L. Hitchcock, The expression of a tensor or a polyadic as asum of products, Studies in Applied Mathematics 6 (1-4) (1927)164–189.

[7] L. R. Tucker, Some mathematical notes on three-mode factoranalysis, Psychometrika 31 (3) (1966) 279–311.

[8] M. H. Van Benthem, M. R. Keenan, Fast algorithm for the solution of large-scale non-negativity-constrained least squares problems, Journal of chemometrics 18 (10) (2004) 441–450.

[9] P. Paatero, A weighted non-negative least squares algorithm for three-way parafacfactor analysis, Chemometrics and Intelligent Laboratory Systems 38 (2) (1997) 223–242.	


[10] Y.-D. Kim, S. Choi, Nonnegative tucker decomposition, in: Computer Vision and Pattern Recognition, 2007. CVPR’07. IEEE Conference on, IEEE, 2007, pp. 1–8.
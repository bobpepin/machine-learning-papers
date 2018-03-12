Interesting Machine Learning Papers
=======================

Fundamentals
=============

A Walk With SGD
-------------
https://arxiv.org/abs/1802.08770

Date: Sat, 24 Feb 2018 00:21:10 GMT   (2201kb,D)

Authors: Chen Xing, Devansh Arpit, Christos Tsirigotis, Yoshua Bengio

Exploring why stochastic gradient descent (SGD) based optimization methods
train deep neural networks (DNNs) that generalize well has become an active
area of research recently. Towards this end, we empirically study the dynamics
of SGD when training over-parametrized deep networks. Specifically we study the
DNN loss surface along the trajectory of SGD by interpolating the loss surface
between parameters from consecutive \textit{iterations} and tracking various
metrics during the training process. We find that the covariance structure of
the noise induced due to mini-batches is quite special that allows SGD to
descend and explore the loss surface while avoiding barriers along its path.
Specifically, our experiments show evidence that for the most part of training,
SGD explores regions along a valley by bouncing off valley walls at a height
above the valley floor. This 'bouncing off walls at a height' mechanism helps
SGD traverse larger distance for small batch sizes and large learning rates
which we find play qualitatively different roles in the dynamics. While a large
learning rate maintains a large height from the valley floor, a small batch
size injects noise facilitating exploration. We find this mechanism is crucial
for generalization because the floor of the valley has barriers and this
exploration above the valley floor allows SGD to quickly travel far away from
the initialization point (without being affected by barriers) and find flatter
regions, corresponding to better generalization.

Train on Validation: Squeezing the Data Lemon
----------------------
https://arxiv.org/abs/1802.05846

Guy Tennenholtz, Tom Zahavy, Shie Mannor
(Submitted on 16 Feb 2018)

Model selection on validation data is an essential step in machine learning. While the mixing of data between training and validation is considered taboo, practitioners often violate it to increase performance. Here, we offer a simple, practical method for using the validation set for training, which allows for a continuous, controlled trade-off between performance and overfitting of model selection. We define the notion of on-average-validation-stable algorithms as one in which using small portions of validation data for training does not overfit the model selection process. We then prove that stable algorithms are also validation stable. Finally, we demonstrate our method on the MNIST and CIFAR-10 datasets using stable algorithms as well as state-of-the-art neural networks. Our results show significant increase in test performance with a minor trade-off in bias admitted to the model selection process.


Generalization in Machine Learning via Analytical Learning Theory
-------------------
https://arxiv.org/abs/1802.07426

Kenji Kawaguchi, Yoshua Bengio

(Submitted on 21 Feb 2018)

This paper introduces a novel measure-theoretic learning theory to analyze generalization behaviors of practical interest. The proposed learning theory has the following abilities: 1) to utilize the qualities of each learned representation on the path from raw inputs to outputs in representation learning, 2) to guarantee good generalization errors possibly with arbitrarily rich hypothesis spaces (e.g., arbitrarily large capacity and Rademacher complexity) and non-stable/non-robust learning algorithms, and 3) to clearly distinguish each individual problem instance from each other. Our generalization bounds are relative to a representation of the data, and hold true even if the representation is learned. We discuss several consequences of our results on deep learning, one-shot learning and curriculum learning. Unlike statistical learning theory, the proposed learning theory analyzes each problem instance individually via measure theory, rather than a set of problem instances via statistics. Because of the differences in the assumptions and the objectives, the proposed learning theory is meant to be complementary to previous learning theory and is not designed to compete with it.


On Nonlinear Dimensionality Reduction, Linear Smoothing and Autoencoding
--------------------------------------
https://arxiv.org/abs/1803.02432

Daniel Ting, Michael I. Jordan

(Submitted on 6 Mar 2018)

We develop theory for nonlinear dimensionality reduction (NLDR). A number of NLDR methods have been developed, but there is limited understanding of how these methods work and the relationships between them. There is limited basis for using existing NLDR theory for deriving new algorithms. We provide a novel framework for analysis of NLDR via a connection to the statistical theory of linear smoothers. This allows us to both understand existing methods and derive new ones. We use this connection to smoothing to show that asymptotically, existing NLDR methods correspond to discrete approximations of the solutions of sets of differential equations given a boundary condition. In particular, we can characterize many existing methods in terms of just three limiting differential operators and boundary conditions. Our theory also provides a way to assert that one method is preferable to another; indeed, we show Local Tangent Space Alignment is superior within a class of methods that assume a global coordinate chart defines an isometric embedding of the manifold.


WNGrad: Learn the Learning Rate in Gradient Descent
---------------------------------------------------
https://arxiv.org/abs/1803.02865

Xiaoxia Wu, Rachel Ward, Léon Bottou

(Submitted on 7 Mar 2018)

Adjusting the learning rate schedule in stochastic gradient methods is an important unresolved problem which requires tuning in practice. If certain parameters of the loss function such as smoothness or strong convexity constants are known, theoretical learning rate schedules can be applied. However, in practice, such parameters are not known, and the loss function of interest is not convex in any case. The recently proposed batch normalization reparametrization is widely adopted in most neural network architectures today because, among other advantages, it is robust to the choice of Lipschitz constant of the gradient in loss function, allowing one to set a large learning rate without worry. Inspired by batch normalization, we propose a general nonlinear update rule for the learning rate in batch and stochastic gradient descent so that the learning rate can be initialized at a high value, and is subsequently decreased according to gradient observations along the way. The proposed method is shown to achieve robustness to the relationship between the learning rate and the Lipschitz constant, and near-optimal convergence rates in both the batch and stochastic settings (O(1/T) for smooth loss in the batch setting, and O(1/T‾‾√) for convex loss in the stochastic setting). We also show through numerical evidence that such robustness of the proposed method extends to highly nonconvex and possibly non-smooth loss function in deep learning problems.Our analysis establishes some first theoretical understanding into the observed robustness for batch normalization and weight normalization.

General AI
==========
A Multi-Objective Deep Reinforcement Learning Framework
-------------------------------------------------------
https://arxiv.org/abs/1803.02965

Thanh Thi Nguyen

(Submitted on 8 Mar 2018)

This paper presents a new multi-objective deep reinforcement learning (MODRL) framework based on deep Q-networks. We propose linear and non-linear methods to develop the MODRL framework that includes both single-policy and multi-policy strategies. The experimental results on a deep sea treasure environment indicate that the proposed approach is able to converge to the optimal Pareto solutions. The proposed framework is generic, which allows implementation of different deep reinforcement learning algorithms in various complex environments. Details of the framework implementation can be referred to this http URL

Theoretical Interest
====================

Averaging Stochastic Gradient Descent on Riemannian Manifolds
----------
https://arxiv.org/abs/1802.09128

Nilesh Tripuraneni, Nicolas Flammarion, Francis Bach, Michael I. Jordan

(Submitted on 26 Feb 2018)

We consider the minimization of a function defined on a Riemannian manifold  accessible only through unbiased estimates of its gradients. We develop a geometric framework to transform a sequence of slowly converging iterates generated from stochastic gradient descent (SGD) on  to an averaged iterate sequence with a robust and fast O(1/n) convergence rate. We then present an application of our framework to geodesically-strongly-convex (and possibly Euclidean non-convex) problems. Finally, we demonstrate how these ideas apply to the case of streaming k-PCA, where we show how to accelerate the slow rate of the randomized power method (without requiring knowledge of the eigengap) into a robust algorithm achieving the optimal rate of convergence.

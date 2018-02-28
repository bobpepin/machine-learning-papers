Interesting Machine Learning Papers
=======================

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


Averaging Stochastic Gradient Descent on Riemannian Manifolds
----------
https://arxiv.org/abs/1802.09128

Nilesh Tripuraneni, Nicolas Flammarion, Francis Bach, Michael I. Jordan

(Submitted on 26 Feb 2018)

We consider the minimization of a function defined on a Riemannian manifold  accessible only through unbiased estimates of its gradients. We develop a geometric framework to transform a sequence of slowly converging iterates generated from stochastic gradient descent (SGD) on  to an averaged iterate sequence with a robust and fast O(1/n) convergence rate. We then present an application of our framework to geodesically-strongly-convex (and possibly Euclidean non-convex) problems. Finally, we demonstrate how these ideas apply to the case of streaming k-PCA, where we show how to accelerate the slow rate of the randomized power method (without requiring knowledge of the eigengap) into a robust algorithm achieving the optimal rate of convergence.

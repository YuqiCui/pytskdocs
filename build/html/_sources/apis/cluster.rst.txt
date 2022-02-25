=======================
API: pytsk.cluster
=======================

:code:`pytsk.cluster` mainly provide fuzzy clustering algorithms. Each algorithm will implemente a :code:`transform` method, which convert the input of raw feature :math:`X \in \mathbb{R}^{N,D}` into the consequent input matrix :math:`P \in \mathbb{R}^{N,T}` of TSK fuzzy systems, where :math:`D` is the input dimension, :math:`N` is the number of samples, for a zero order TSK fuzzy system, :math:`T=R`, where :math:`R` is the number of rules (equal to the number of clusters of the fuzzy clustering algorithm), for a first-order TSK fuzzy system, :math:`T = (D+1)\times R`.

.. py:class:: pytsk.cluster.BaseFuzzyClustering()

    Parent: :code:`object`.

    The parent class of fuzzy clustering classes.

    .. py:method:: set_params(self, **params)

        Setting attributes. Implemented to adapt the API of scikit-learn.

.. py:class:: pytsk.cluster.FuzzyCMeans(n_cluster, fuzzy_index="auto", sigma_scale="auto", init="random", tol_iter=100, error=1e-6, dist="euclidean", verbose=0, order=1)

    Parent: :func:`BaseFuzzyClustering <pytsk.cluster.BaseFuzzyClustering>`, :func:`sklearn.base.BaseEstimator <https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html>`, :func:`sklearn.base.TransformerMixin <https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html>`.

    The fuzzy c-means (FCM) clustering algorithm [1]. This implementation is adopted from the `scikit-fuzzy <https://pythonhosted.org/scikit-fuzzy/overview.html>`_ package. When constructing a TSK fuzzy system, a fuzzy clustering algorithm is usually used to compute the antecedent parameters, after that, the consequent parameters can be computed by least-squared error algorithms, such as Ridge regression [2]. How to use this class can be found at `Quick start <quick_start.html#training-with-fuzzy-clustering>`_.

    The objective function of the FCM is:

    .. math::
        &J = \sum_{i=1}^{N}\sum_{j=1}^{C} U_{i,j}^m\|\mathbf{x}_i - \mathbf{v}_j\|_2^2\\
        &s.t. \sum_{j=1}^{C}\mu_{i,j} = 1, i = 1,...,N,

    where :math:`N` is the number of samples, :math:`C` is the number of clusters (which also corresponding to the number of rules of TSK fuzzy systems), :math:`m` is the fuzzy index, :math:`\mathbf{x}_i` is the :math:`i`-th input vector, :math:`\mathbf{v}_j` is the :math:`j`-th cluster center vector, :math:`U_{i,j}` is the membership degree of the :math:`i`-th input vector on the :math:`j`-th cluster center vector. The FCM algorithm will obtain the centers  :math:`\mathbf{v}_j, j=1,...,C` and the membership degrees :math:`U_{i,j}`.

    :param int n_cluster: Number of clusters, equal to the number of rules :math:`R` of a TSK model.
    :param float/str fuzzy_index: Fuzzy index of the FCM algorithm, default `auto`. If :code:`fuzzy_index=auto`, then the fuzzy index is computed as :math:`\min(N, D-1) / (\min(N, D-1)-2)` (If :math:`\min(N, D-1)<3`, fuzzy index will be set to 2), according to [3]. Otherwise the given float value is used.
    :param float/str sigma_scale: The scale parameter :math:`h` to adjust the actual standard deviation :math:`\sigma` of the Gaussian membership function in TSK antecedent part. If :code:`sigma_scale=auto`, :code:`sigma_scale` will be set as :math:`\sqrt{D}`, where :math:`D` is the input dimension [4]. Otherwise the given float value is used.
    :param str/np.array init: The initialization strategy of the membership grid matrix :math:`U`. Support "random" or numpy array with the size of :math:`[R, N]`, where :math:`R` is the number of clusters/rules, :math:`N` is the number of training samples. If :code:`init="random"`, the initial membership grid matrix will be randomly initialized, otherwise the given matrix will be used.
    :param int tol_iter: The total iteration of the FCM algorithm.
    :param float error: The maximum error that will stop the iteration before maximum iteration is reached.
    :param str dist: The distance type for the :func:`scipy.spatial.distance.cdist` function, default "euclidean". The distance function can also be "braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean", "hamming", "jaccard", "jensenshannon", "kulsinski", "kulczynski1", "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule".
    :param int verbose: If > 0, it will show the loss of the FCM objective function during iterations.
    :param int order: 0 or 1. Decide whether to construct a zero-order TSK or a first-order TSK.

    .. py:method:: fit(self, X, y=None)

        Run the FCM algorithm.

        :param numpy.array X: Input array with the size of :math:`[N, D]`, where :math:`N` is the number of training samples, and :math:`D` is number of features.
        :param numpy.array y: Not used. Pass None.

    .. py:method:: predict(self, X, y=None)

        Predict the membership degrees of :code:`X` on each cluster.

        :param numpy.array X: Input array with the size of :math:`[N, D]`, where :math:`N` is the number of training samples, and :math:`D` is number of features.
        :param numpy.array y: Not used. Pass None.
        :return: return the membership degree matrix :math:`U` with the size of :math:`[N, R]`, where :math:`N` is the number of samples of :code:`X`, and :math:`R` is the number of clusters/rules. :math:`U_{i,j}` represents the membership degree of the :math:`i`-th sample on the :math:`r`-th cluster.

    .. py:method:: transform(self, X, y=None)

        Compute the membership degree matrix :math:`U`, and use :math:`X` and :math:`U` to get the consequent input matrix :math:`P` using function :func:`x2xp(x, u, order) <x2xp>`

        :param numpy.array X: Input array with the size of :math:`[N, D]`, where :math:`N` is the number of training samples, and :math:`D` is number of features.
        :param numpy.array y: Not used. Pass None.
        :return: return the consequent input :math:`X_p` with the size of :math:`[N, (D+1)\times R]`, where :math:`N` is the number of test samples, :math:`D` is number of features, :math:`R` is the number of clusters/rules.

.. py:function:: x2xp(X, U, order)

    Convert the feature matrix :math:`X` and the membership degree matrix :math:`U` into the consequent input matrix :math:`X_p`

    Each row in :math:`X\in \mathbb{R}^{N,D}` represents a :math:`D`-dimension input vector. Suppose vector :math:`\mathbf{x}` is one row, and then the consequent input matrix :math:`P` is computed as [5] for a first-order TSK:

    .. math::
        &\mathbf{x}_e = (1, \mathbf{x}),\\
        &\tilde{\mathbf{x}}_r = u_r \mathbf{x}_e,\\
        &\mathbf{p} = (\tilde{\mathbf{x}}_1, \tilde{\mathbf{x}}_2, ...,\tilde{\mathbf{x}}_R),

    where :math:`\mathbf{p}` is the corresponding row in :math:`P`, which is a :math:`(D+1)\times R`-dimension vector. Then the consequent parameters of TSK can be optimized by any linear regression algorithms.


    :param numpy.array x: size: :math:`[N,D]`. Input features.
    :param numpy.array u: size: :math:`[N,R]`. Corresponding membership degree matrix.
    :param int order: 0 or 1. The order of TSK models.
    :return: If :code:`order=0`, return :math:`U` directly, else if :code:`order=1`, return the matrix :math:`X_p` with the size of :math:`[N, (D+1)\times R]`. Details can be found at [2].


.. py:function:: compute_variance(X, U, V)

    Compute the variance of the Gaussian membership function in TSK fuzzy systems. After performing the FCM, one can use :math:`\mathbf{v}_j` and :math:`U_{i,j}` to construct the Gaussian membership function based antecedent of a TSK fuzzy system. The center of the Gaussian membership function can be directly set as center `\mathbf{v}_j`, the standard deviation of the Gaussian membership function can be computed as follows:

        .. math::
            \sigma_{r,d}=\left[\sum_{i=1}^N U_{i,r}(x_{i,d}-v_{r,d})^2 / \sum_{i=1}^N U_{i,r} \right]^{1/2},

        where :math:`v_{r,d}` represents the cluster center of the :math:`d`-th dimension in the :math:`r`-th rule.

    :param numpy.array x: Input matrix :math:`X` with the size of :math:`[N, D]`.
    :param numpy.array u: Membership degree matrix :math:`U` with the size of :math:`[R, N]`.
    :param numpy.array v: Cluster center matrix :math:`V` with the size of :math:`[R, D]`.
    :return: The standard variation matrix :math:`\Sigma` with the size of :math:`[R, D]`.

    [1] `Bezdek J C, Ehrlich R, Full W. FCM: The fuzzy c-means clustering algorithm[J]. Computers & geosciences, 1984, 10(2-3): 191-203. <https://www.sciencedirect.com/science/article/pii/0098300484900207>`_

    [2] `Wang S, Chung K F L, Zhaohong D, et al. Robust fuzzy clustering neural network based on ɛ-insensitive loss function[J]. Applied Soft Computing, 2007, 7(2): 577-584. <https://www.sciencedirect.com/science/article/pii/S1568494606000469>`_

    [3] `Yu J, Cheng Q, Huang H. Analysis of the weighting exponent in the FCM[J]. IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 2004, 34(1): 634-639. <https://ieeexplore.ieee.org/abstract/document/1262532/>`_

    [4] `Cui Y, Wu D, Xu Y. Curse of dimensionality for tsk fuzzy neural networks: Explanation and solutions[C]//2021 International Joint Conference on Neural Networks (IJCNN). IEEE, 2021: 1-8. <https://arxiv.org/pdf/2102.04271.pdf>`_

    [5] `Deng Z, Choi K S, Chung F L, et al. Scalable TSK fuzzy modeling for very large datasets using minimal-enclosing-ball approximation[J]. IEEE Transactions on Fuzzy Systems, 2010, 19(2): 210-226. <https://ieeexplore.ieee.org/abstract/document/5629439/>`_


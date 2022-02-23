============================
API: pytsk.gradient_descent
============================

This package contains all the APIs you need to build and train a fuzzy neural networks.

pytsk.gradient_descent.antecedent
####################################

.. py:class:: AntecedentGMF(in_dim, n_rule, high_dim=False, init_center=None, init_sigma=1., eps=1e-8)

    Parent: :code:`torch.nn.Module`.

    The antecedent part with Gaussian membership function. Input: data, output the corresponding firing levels of each rule.

    :param int in_dim: Number of features :math:`D` of the input.
    :param int n_rule: Number of rules :math:`R` of the TSK model.
    :param bool high_dim: Whether to use the HTSK defuzzification. If :code:`high_dim=True`, HTSK is used. Otherwise the original defuzzification is used. More details can be found at [1]. TSK model tends to fail on high-dimensional problems, so set :code:`high_dim=True` is highly recommended for any-dimensional problems.
    :param numpy.array init_center: Initial center of the Gaussian membership function with the size of :math:`[D,R]`. A common way is to run a KMeans clustering and set :code:`init_center` as the obtained centers. You can simply run :func:`pytsk.gradient_descent.antecedent.antecedent_init_center <antecedent_init_center>` to obtain the center.
    :param float init_sigma: Initial :math:`\sigma` of the Gaussian membership function.
    :param float eps: A constant to avoid the division zero error.

    .. py:method:: init(self, center, sigma)

        Change the value of :code:`init_center` and :code:`init_sigma`.

        :param numpy.array center: Initial center of the Gaussian membership function with the size of :math:`[D,R]`. A common way is to run a KMeans clustering and set :code:`init_center` as the obtained centers. You can simply run :func:`pytsk.gradient_descent.antecedent.antecedent_init_center <antecedent_init_center>` to obtain the center.
        :param float sigma: Initial :math:`\sigma` of the Gaussian membership function.

    .. py:method:: reset_parameters(self)

        Re-initialize all parameters.

    .. py:method:: forward(self, X)

        Forward method of Pytorch Module.

        :param torch.tensor X: pytorch tensor with the size of :math:`[N, D]`, where :math:`N` is the number of samples, :math:`D` is the input dimension.
        :return: Firing level matrix :math:`U` with the size of :math:`[N, R]`.


.. py:function:: antecedent_init_center(X, y=None, n_rule=2, method="kmean", engine="sklearn", n_init=20)

[1] `Cui Y, Wu D, Xu Y. Curse of dimensionality for tsk fuzzy neural networks: Explanation and solutions[C]//2021 International Joint Conference on Neural Networks (IJCNN). IEEE, 2021: 1-8. <https://arxiv.org/pdf/2102.04271.pdf>`_

pytsk.gradient_descent.callbacks
####################################

pytsk.gradient_descent.training
####################################

pytsk.gradient_descent.tsk
####################################

pytsk.gradient_descent.utils
####################################


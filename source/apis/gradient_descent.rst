============================
API: pytsk.gradient_descent
============================

This package contains all the APIs you need to build and train a fuzzy neural networks.

pytsk.gradient_descent.antecedent
####################################

.. py:class:: AntecedentGMF(in_dim, n_rule, high_dim=False, init_center=None, init_sigma=1., eps=1e-8)

    Parent: :code:`torch.nn.Module`

    The antecedent part with Gaussian membership function. Input: data, output the corresponding firing levels of each rule. The firing level :math:`f_r(\mathbf{x})` of the :math:`r`-th rule are computed by:

    .. math::
        &\mu_{r,d}(x_d) = \exp(-\frac{(x_d - m_{r,d})^2}{2\sigma_{r,d}^2}),\\
        &f_{r}(\mathbf{x})=\prod_{d=1}^{D}\mu_{r,d}(x_d),\\
        &\overline{f}_r(\mathbf{x}) = \frac{f_{r}(\mathbf{x})}{\sum_{i=1}^R f_{i}(\mathbf{x})}.


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

        :param torch.tensor X: Pytorch tensor with the size of :math:`[N, D]`, where :math:`N` is the number of samples, :math:`D` is the input dimension.
        :return: Firing level matrix :math:`U` with the size of :math:`[N, R]`.


.. py:class:: AntecedentShareGMF(in_dim, n_mf=2, high_dim=False, init_center=None, init_sigma=1., eps=1e-8)

    Parent: :code:`torch.nn.Module`

    The antecedent part with Gaussian membership function, rules will share the membership functions on each feature [2]. The number of rules will be :math:`M^D`, where :math:`M` is :code:`n_mf`, :math:`D` is the number of features (:code:`in_dim`).

    :param int in_dim: Number of features :math:`D` of the input.
    :param int n_mf: Number of membership functions :math:`M` of each feature.
    :param bool high_dim: Whether to use the HTSK defuzzification. If :code:`high_dim=True`, HTSK is used. Otherwise the original defuzzification is used. More details can be found at [1]. TSK model tends to fail on high-dimensional problems, so set :code:`high_dim=True` is highly recommended for any-dimensional problems.
    :param numpy.array init_center: Initial center of the Gaussian membership function with the size of :math:`[D,M]`.
    :param float init_sigma: Initial :math:`\sigma` of the Gaussian membership function.
    :param float eps: A constant to avoid the division zero error.

    .. py:method:: init(self, center, sigma)

        Change the value of :code:`init_center` and :code:`init_sigma`.

        :param numpy.array center: Initial center of the Gaussian membership function with the size of :math:`[D,M]`.
        :param float sigma: Initial :math:`\sigma` of the Gaussian membership function.

    .. py:method:: reset_parameters(self)

        Re-initialize all parameters.

    .. py:method:: forward(self, X)

        Forward method of Pytorch Module.

        :param torch.tensor X: Pytorch tensor with the size of :math:`[N, D]`, where :math:`N` is the number of samples, :math:`D` is the input dimension.
        :return: Firing level matrix :math:`U` with the size of :math:`[N, R], R=M^D`:.

.. py:function:: antecedent_init_center(X, y=None, n_rule=2, method="kmean", engine="sklearn", n_init=20)

    This function run KMeans clustering to obtain the :code:`init_center` for :func:`AntecedentGMF() <AntecedentGMF>`.

    Examples
    --------
    >>> init_center = antecedent_init_center(X, n_rule=10, method="kmean", n_init=20)
    >>> antecedent = AntecedentGMF(X.shape[1], n_rule=10, init_center=init_center)

    :param numpy.array X: Feature matrix with the size of :math:`[N,D]`, where :math:`N` is the number of samples, :math:`D` is the number of features.
    :param numpy.array y: None, not used.
    :param int n_rule: Number of rules :math:`R`. This function will run a KMeans clustering to obtain :math:`R` cluster centers as the initial antecedent center for TSK modeling.
    :param str method: Current version only support "kmean".
    :param str engine: "sklearn" or "faiss". If "sklearn", then the :code:`sklearn.cluster.KMeans()` function will be used, otherwise the :code:`faiss.Kmeans()` will be used. Faiss provide a faster KMeans clustering algorithm, "faiss" is recommended for large datasets.
    :param int n_init: Number of initialization of the KMeans algorithm. Same as the parameter :code:`n_init` in :code:`sklearn.cluster.KMeans()` and the parameter :code:`nredo` in :code:`faiss.Kmeans()`.

[1] `Cui Y, Wu D, Xu Y. Curse of dimensionality for tsk fuzzy neural networks: Explanation and solutions[C]//2021 International Joint Conference on Neural Networks (IJCNN). IEEE, 2021: 1-8. <https://arxiv.org/pdf/2102.04271.pdf>`_

[2] `Shi Y, Mizumoto M. A new approach of neuro-fuzzy learning algorithm for tuning fuzzy rules[J]. Fuzzy sets and systems, 2000, 112(1): 99-116. <https://www.sciencedirect.com/science/article/pii/S0165011498002383>`_

pytsk.gradient_descent.tsk
####################################

.. py:class:: TSK(in_dim, out_dim, n_rule, antecedent, order=1, eps=1e-8, precons=None)

    Parent: :code:`torch.nn.Module`

    This module define the consequent part of the TSK model and combines it with a pre-defined antecedent module. The input of this module is the raw feature matrix, and output the final prediction of a TSK model.

    :param int in_dim: Number of features :math:`D`.
    :param int out_dim: Number of output dimension :math:`C`.
    :param int n_rule: Number of rules :math:`R`, must equal to the :code:`n_rule` of the :code:`Antecedent()`.
    :param torch.Module antecedent: An antecedent module, whose output dimension should be equal to the number of rules :math:`R`.
    :param int order: 0 or 1. The order of TSK. If 0, zero-oder TSK, else, first-order TSK.
    :param float eps: A constant to avoid the division zero error.
    :param torch.nn.Module consbn: If none, the raw feature will be used as the consequent input; If a pytorch module, then the consequent input will be the output of the given module. If you wish to use the BN technique we mentioned in `Models & Technique <../models.html#batch-normalization>`_, you can set :code:`precons=nn.BatchNorm1d(in_dim)`.

    .. py:method:: reset_parameters(self)

        Re-initialize all parameters, including both consequent and antecedent parts.

    .. py:method:: forward(self, X, get_frs=False)

        :param torch.tensor X: Input matrix with the size of :math:`[N, D]`, where :math:`N` is the number of samples.
        :param bool get_frs: If true, the firing levels (the output of the antecedent) will also be returned.

        :return: If :code:`get_frs=True`, return the TSK output :math:`Y\in \mathbb{R}^{N,C}` and the antecedent output :math:`U\in \mathbb{R}^{N,R}`. If :code:`get_frs=False`,  only return the TSK output :math:`Y`.

pytsk.gradient_descent.training
####################################

.. py:function:: ur_loss(frs, tau=0.5)

    The uniform regularization (UR) proposed by Cui et al. [3]. UR loss is computed as :math:`\ell_{UR} = \sum_{r=1}^R (\frac{1}{N}\sum_{n=1}^N f_{n,r} - \tau)^2`,
    where :math:`f_{n,r}` represents the firing level of the :math:`n`-th sample on the :math:`r`-th rule.

    :param torch.tensor frs: The firing levels (output of the antecedent) with the size of :math:`[N, R]`, where :math:`N` is the number of samples, :math:`R` is the number of ruels.
    :param float tau: The expectation :math:`\tau` of the average firing level for each rule. For a :math:`C`-class classification problem, we recommend setting :math:`\tau` to :math:`1/C`, for a regression problem, :math:`\tau` can be set as :math:`0.5`.
    :return: A scale value, representing the UR loss.


.. py:class:: Wrapper(model, optimizer, criterion, batch_size=512, epochs=1, callbacks=None, label_type="c", device="cpu", reset_param=True, ur=0, ur_tau=0.5, **kwargs)

    This class provide a training framework for beginners to train their fuzzy neural networks.

    :param torch.nn.Module model: The pre-defined TSK model.
    :param torch.Optimizer optimizer: Pytorch optimizer.
    :param torch.nn._Loss: Pytorch loss. For example, :code:`torch.nn.CrossEntropyLoss()` for classification tasks, and :code:`torch.nn.MSELoss()` for regression tasks.
    :param int batch_size: Batch size during training & prediction.
    :param int epochs: Training epochs.
    :param [Callback] callbacks: List of callbacks.
    :param str label_type: Label type, "c" or "r", when :code:`label_type="c"`, label's dtype will be changed to "int64", when :code:`label_type="r"`, label's dtype will be changed to "float32".

    Examples
    --------
    >>> from pytsk.gradient_descent import antecedent_init_center, AntecedentGMF, TSK, EarlyStoppingACC, EvaluateAcc, Wrapper
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import accuracy_score
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.preprocessing import StandardScaler
    >>> from torch.optim import AdamW
    >>> import torch.nn as nn
    >>> # ----------------- define data -----------------
    >>> X, y = make_classification(random_state=0)
    >>> x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    >>> ss = StandardScaler()
    >>> x_train = ss.fit_transform(x_train)
    >>> x_test = ss.transform(x_test)
    >>> # ----------------- define TSK model -----------------
    >>> n_rule = 10  # define number of rules
    >>> n_class = 2  # define output dimension
    >>> order = 1  # first-order TSK is used
    >>> consbn = True  # consbn tech is used
    >>> weight_decay = 1e-8  # weight decay for pytorch optimizer
    >>> lr = 0.01  # learning rate for pytorch optimizer
    >>> init_center = antecedent_init_center(x_train, y_train, n_rule=n_rule)  # obtain the initial antecedent center
    >>> gmf = AntecedentGMF(in_dim=x_train.shape[1], n_rule=n_rule, high_dim=True, init_center=init_center)  # define antecedent
    >>>  model = TSK(in_dim=x_train.shape[1], out_dim=n_class, n_rule=n_rule, antecedent=gmf, order=order, consbn=consbn)  # define TSK
    >>> # ----------------- define optimizers -----------------
    >>> ante_param, other_param = [], []
    >>> for n, p in model.named_parameters():
    >>>     if "center" in n or "sigma" in n:
    >>>         ante_param.append(p)
    >>>     else:
    >>>         other_param.append(p)
    >>> optimizer = AdamW(
    >>>     [{'params': ante_param, "weight_decay": 0},  # antecedent parameters usually don't need weight_decay
    >>>     {'params': other_param, "weight_decay": weight_decay},],
    >>>     lr=lr
    >>> )
    >>> # ----------------- split 20% data for earlystopping -----------------
    >>> x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
    >>> # ----------------- define the earlystopping callback -----------------
    >>> EACC = EarlyStoppingACC(x_val, y_val, verbose=1, patience=40, save_path="tmp.pkl")  # Earlystopping
    >>> TACC = EvaluateAcc(x_test, y_test, verbose=1)  # Check test acc during training
    >>> # ----------------- train model -----------------
    >>> wrapper = Wrapper(model, optimizer=optimizer, criterion=nn.CrossEntropyLoss(),
    >>>               epochs=300, callbacks=[EACC, TACC], ur=0, ur_tau=1/n_class)  # define training wrapper, ur weight is set to 0
    >>> wrapper.fit(x_train, y_train)  # fit
    >>> wrapper.load("tmp.pkl")  # load best model saved by EarlyStoppingACC callback
    >>> y_pred = wrapper.predict(x_test).argmax(axis=1)  # predict, argmax for extracting classfication label
    >>> print("[TSK] ACC: {:.4f}".format(accuracy_score(y_test, y_pred)))  # print ACC

    .. py:method:: train_on_batch(self, input, target)

        Define how to update a model with one batch of data. This method can be overwrite for custom training strategy.

        :param torch.tensor input: Feature matrix with the size of :math:`[N,D]`, :math:`N` is the number of samples, :math:`D` is the input dimension.
        :param torch.tensor target: Target matrix with the size of :math:`[N,C]`, :math:`C` is the output dimension.

    .. py:method:: fit(X, y)

        Train the :code:`model` with numpy array.

        :param numpy.array X: Feature matrix :math:`X` with the size of :math:`[N, D]`.
        :param numpy.array y: Label matrix :math:`Y` with the size of :math:`[N, C]`, for classification task, :math:`C=1`, for regression task, :math:`C` is the number of the output dimension of :code:`model`.

    .. py:method:: fit_loader(self, train_loader)

        Train the :code:`model` with user-defined pytorch dataloader.

        :param torch.utils.data.DataLoader train_loader: Data loader, the output of the loader should be corresponding to the inputs of :func:`train_on_batch <train_on_batch>`. For example, if dataloader has two output, then :func:`train_on_batch <train_on_batch>` should also have two inputs.

    .. py:method:: predict(self, X, y=None)

        Get the prediction of the model.

        :param numpy.array X: Feature matrix :math:`X` with the size of :math:`[N, D]`.
        :param y: Not used.
        :return: Prediction matrix :math:`\hat{Y}` with the size of :math:`[N, C]`, :math:`C` is the output dimension of the :code:`model`.

    .. py:method:: predict_proba(self, X, y=None)

        For classification problem only, need :code:`label_type="c"`, return the prediction after softmax.

        :param numpy.array X: Feature matrix :math:`X` with the size of :math:`[N, D]`.
        :param y: Not used.
        :return: Prediction matrix :math:`\hat{Y}` with the size of :math:`[N, C]`, :math:`C` is the output dimension of the :code:`model`.

    .. py:method:: save(self, path)

        Save model.

        :param str path: Model save path.

    .. py:method:: load(self, path)

        Load model.

        :param str path: Model save path.


[3] `Cui Y, Wu D, Huang J. Optimize tsk fuzzy systems for classification problems: Minibatch gradient descent with uniform regularization and batch normalization[J]. IEEE Transactions on Fuzzy Systems, 2020, 28(12): 3065-3075. <https://ieeexplore.ieee.org/abstract/document/8962207/>`_

pytsk.gradient_descent.callbacks
####################################


.. py:class:: Callback()

    Similar as the callback class in Keras, our package provides a simplified version of callback, which allow users to monitor metrics during the training. We strongly recommend uses to custom their callbacks, here we provide two examples, :func:`EvaluateAcc <EvaluateAcc>` and :func:`EarlyStoppingACC <EarlyStoppingACC>`.

    .. py:method:: on_batch_begin(self, wrapper)

        Will be called before each batch.

    .. py:method:: on_batch_end(self, wrapper)

        Will be called after each batch.

    .. py:method:: on_epoch_begin(self, wrapper)

        Will be called before each epoch.

    .. py:method:: on_epoch_end(self, wrapper)

        Will be called after each epoch.


.. py:class:: EvaluateAcc(X, y, verbose=0)

    Evaluate the accuracy during training.

    :param numpy.array X: Feature matrix with the size of :math:`[N, D]`.
    :param numpy.array y: Label matrix with the size of :math:`[N, 1]`.

    .. py:method:: on_epoch_end(self, wrapper)

    Examples
    --------
    >>> def on_epoch_end(self, wrapper):
    >>>     cur_log = {}
    >>>     y_pred = wrapper.predict(self.X).argmax(axis=1)
    >>>     acc = accuracy_score(y_true=self.y, y_pred=y_pred)
    >>>     cur_log["epoch"] = wrapper.cur_epoch
    >>>     cur_log["acc"] = acc
    >>>     self.logs.append(cur_log)
    >>>     if self.verbose > 0:
    >>>         print("[Epoch {:5d}] Test ACC: {:.4f}".format(cur_log["epoch"], cur_log["acc"]))


    :param Wrapper wrapper: The training :func:`Wrapper <Wrapper>`.


.. py:class:: EarlyStoppingACC(X, y, patience=1, verbose=0, save_path=None)

    Early-stopping by classification accuracy.

    :param numpy.array X: Feature matrix with the size of :math:`[N, D]`.
    :param numpy.array y: Label matrix with the size of :math:`[N, 1]`.
    :param int patience: Number of epochs with no improvement after which training will be stopped.
    :param int verbose: verbosity mode.
    :param str save_path: If :code:`save_path=None`, do not save models, else save the model with the best accuracy to the given path.

    .. py:method:: on_epoch_end(self, wrapper)

        Calculate the validation accuracy and determine whether to stop training.

        Examples
        --------
        >>> def on_epoch_end(self, wrapper):
        >>>     cur_log = {}
        >>>     y_pred = wrapper.predict(self.X).argmax(axis=1)
        >>>     acc = accuracy_score(y_true=self.y, y_pred=y_pred)
        >>>     if acc > self.best_acc:
        >>>         self.best_acc = acc
        >>>         self.cnt = 0
        >>>         if self.save_path is not None:
        >>>             wrapper.save(self.save_path)
        >>>     else:
        >>>         self.cnt += 1
        >>>         if self.cnt > self.patience:
        >>>              wrapper.stop_training = True
        >>>     cur_log["epoch"] = wrapper.cur_epoch
        >>>     cur_log["acc"] = acc
        >>>     cur_log["best_acc"] = self.best_acc
        >>>     self.logs.append(cur_log)
        >>>     if self.verbose > 0:
        >>>         print("[Epoch {:5d}] EarlyStopping Callback ACC: {:.4f}, Best ACC: {:.4f}".format(cur_log["epoch"], cur_log["acc"], cur_log["best_acc"]))

        :param Wrapper wrapper:  The training :func:`Wrapper <Wrapper>`.


pytsk.gradient_descent.utils
####################################

.. py:function:: check_tensor(tensor, dtype)

    Convert :code:`tensor` into a :code:`dtype` torch.Tensor.

    :param numpy.array/torch.tensor tensor: Input data.
    :param str dtype: PyTorch dtype string.
    :return: A :code:`dtype` torch.Tensor.


.. py:function:: reset_params(model)

    Reset all parameters in :code:`model`.

    :param torch.nn.Module model: Pytorch model.


.. py:class:: NumpyDataLoader(*inputs)

    Convert numpy arrays into a dataloader.

    :param numpy.array inputs: Numpy arrays.

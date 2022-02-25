Models & Technique
==============================

TSK
#############################

A basic TSK fuzzy system is a combination of :math:`R` rules, the :math:`r`-th rule can be represented as:

:math:`\text{Rule}_r:~\text{IF}~x_1~\text{is}~X_{r,1}~\text{and}~ ... ~\text{and}~x_D~\text{is}~ X_{r,D}\\ ~~~~~~~~~~~~~\text{THEN}~ y=w_1x_1 + ... + w_Dx_D + b,`

where :math:`x_d` is :math:`d`-th input feature, :math:`X_{r,d}` is the membership function of the :math:`d`-th input feature in the :math:`r`-th rule. The IF part is called antecedent, the THEN part is called consequent in this package. The antecedent output the firing levels of the rules (for those who are not familiar with fuzzy systems, you can understand the firing levels as the attention weight of a Transformer/Mixture-of-experts(MoE) model), and the consequent part output the final prediction.

To define a TSK model, we need to define both antecedent and consequent modules::

    # --------- Data format ------------
    # X: feature matrix, [n_data, n_dim] each row represents a sample with n_dim features
    # y: label matrix, [n_data, 1]

    # --------- Define TSK model parameters ------------
    n_rule = 10 # define num. of rules
    n_class = 2  # define num. of class (model output dimension)
    order = 1  # 0 or 1, zero-order TSK model or first-order TSK model

    # --------- Define antecedent ------------
    # run kmeans clustering to get initial rule centers
    init_center = antecedent_init_center(X, y, n_rule=n_rule)
    # define the antecedent Module
    gmf = AntecedentGMF(in_dim=X.shape[1], n_rule=n_rule, init_center=init_center)

    # --------- Define full TSK model ------------
    model = TSK(in_dim=X.shape[1], out_dim=n_class, n_rule=n_rule, antecedent=gmf, order=order)

HTSK
#############################
Traditional TSK model tends to fail on high-dimensional problems, so **the HTSK (high-dimensional TSK) model is recommended for handling any-dimension problems**. More details about the HTSK model can be found in [1].

To define a HTSK model, we need to set :code:`high_dim=True` when define antecedent::

    init_center = antecedent_init_center(X, y, n_rule=n_rule)
    gmf = AntecedentGMF(in_dim=X.shape[1], n_rule=n_rule, init_center=init_center, high_dim=True)

[1] `Cui Y, Wu D, Xu Y. Curse of dimensionality for tsk fuzzy neural networks: Explanation and solutions[C]//2021 International Joint Conference on Neural Networks (IJCNN). IEEE, 2021: 1-8. <https://arxiv.org/pdf/2102.04271.pdf>`_

DropRule
##############################
Similar as Dropout, randomly dropping rules of TSK (DropRule) can improve the performance of TSK models [2,3,4].

To use DropRule, we need to add a Dropout layer after the antecedent output::

    # --------- Define antecedent ------------
    init_center = antecedent_init_center(X, y, n_rule=n_rule)
    gmf = nn.Sequential(
        AntecedentGMF(in_dim=X.shape[1], n_rule=n_rule, high_dim=True, init_center=init_center),
        nn.Dropout(p=0.25)
    )

    # --------- Define full TSK model ------------
    model = TSK(in_dim=X.shape[1], out_dim=n_class, n_rule=n_rule, antecedent=gmf, order=order)

[2] `Wu D, Yuan Y, Huang J, et al. Optimize TSK fuzzy systems for regression problems: Minibatch gradient descent with regularization, DropRule, and AdaBound (MBGD-RDA)[J]. IEEE Transactions on Fuzzy Systems, 2019, 28(5): 1003-1015. <https://ieeexplore.ieee.org/abstract/document/8930057/>`_

[3] `Shi Z, Wu D, Guo C, et al. FCM-RDpA: tsk fuzzy regression model construction using fuzzy c-means clustering, regularization, droprule, and powerball adabelief[J]. Information Sciences, 2021, 574: 490-504. <https://www.sciencedirect.com/science/article/pii/S0020025521005776>`_

[4] `Guo F, Liu J, Li M, et al. A Concise TSK Fuzzy Ensemble Classifier Integrating Dropout and Bagging for High-dimensional Problems[J]. IEEE Transactions on Fuzzy Systems, 2021. <https://ieeexplore.ieee.org/abstract/document/9520250/>`_

Batch Normalization
###############################
Batch normalization (BN) can be used to normalize the input of consequent parameters, and the experiments in [5] have shown that BN can speed up the convergence and improve the performance of a TSK model.

To add the BN layer, we need to set :code:`precons=nn.BatchNorm1d(in_dim)` when defining the TSK model::

    model = TSK(in_dim=X.shape[1], out_dim=n_class, n_rule=n_rule, antecedent=gmf, order=order, precons=nn.BatchNorm1d(in_dim))

[5] `Cui Y, Wu D, Huang J. Optimize tsk fuzzy systems for classification problems: Minibatch gradient descent with uniform regularization and batch normalization[J]. IEEE Transactions on Fuzzy Systems, 2020, 28(12): 3065-3075. <https://ieeexplore.ieee.org/abstract/document/8962207/>`_

Uniform Regularization
####################################
[5] also proposed a uniform regularization, which can mitigate the "winner gets all" problem when training TSK with mini-batch gradient descent algorithms. The "winner gets all" problem will cause only a small number of rules dominant the prediction, other rules will have nearly zero contribution to the prediction. The uniform regularization loss is:

.. math::
    \ell_{UR} = \sum_{r=1}^R (\frac{1}{N}\sum_{n=1}^N f_{n,r} - \tau)^2,

where :math:`f_{n,r}` represents the firing level of the :math:`n`-th sample on the :math:`r`-th rule, :math:`N` is the batch size. Experiments in [5] has proved that UR can significantly improve the performance of TSK fuzzy system. The UR loss is defined at :func:`ur_loss <ur_loss>`. If you want to use the UR during training, you can simply set a positive UR weight when initialize :func:`Wrapper <Wrapper>`::

    ur = 1.  # must > 0
    ur_tau  # a float number between 0 and 1
    wrapper = Wrapper(
        model, optimizer=optimizer, criterion=criterion, epochs=epochs, callbacks=callbacks, ur=ur, ur_tau=ur_tau
    )

[5] `Cui Y, Wu D, Huang J. Optimize tsk fuzzy systems for classification problems: Minibatch gradient descent with uniform regularization and batch normalization[J]. IEEE Transactions on Fuzzy Systems, 2020, 28(12): 3065-3075. <https://ieeexplore.ieee.org/abstract/document/8962207/>`_

Layer Normalization
###############################
Layer normalization (LN) can be used to normalize the firing levels of the antecedent part [6]. It can be easily proved that the scale of firing levels will decrease when more rules are used. Since the gradient of the parameters in TSK are all relevant with the firing level, it will cause a gradient vanishing problem, making TSKs perform bad, especially when using SGD/SGDM as optimizer. LN normalizes the firing level, similar as the LN layer in Transformer, can solve the gradient vanishing problems and improve the performance. Adding a ReLU acitivation can further filter the negative firing levels generated by LN, improving the interpretability and robustness to outliers.

To add LN & ReLU, we can do::

    # --------- Define antecedent ------------
    init_center = antecedent_init_center(X, y, n_rule=n_rule)
    gmf = nn.Sequential(
        AntecedentGMF(in_dim=X.shape[1], n_rule=n_rule, high_dim=True, init_center=init_center),
        nn.LayerNorm(n_rule),
        nn.ReLU()
    )

    # --------- Define full TSK model ------------
    model = TSK(in_dim=x_train.shape[1], out_dim=n_class, n_rule=n_rule, antecedent=gmf, order=order)

[6] `Cui Y, Wu D, Xu Y, Peng R. Layer Normalization for TSK Fuzzy System Optimization in Regression Problems[J]. IEEE Transactions on Fuzzy Systems, submitted.`

Deep learning
###########################
TSK models can also be used as a classifier/regressor in a deep neural network, which may improving the performance of neural networks. To do that, we first need to get the middle output of neural networks for antecedent initialization, and then define the deep fuzzy systems as follows::

    # --------- Define antecedent ------------
    # Note that X should be the output of NeuralNetworks, y is still the corresponding label
    init_center = antecedent_init_center(X, y, n_rule=n_rule)
    gmf = AntecedentGMF(in_dim=X.shape[1], n_rule=n_rule, high_dim=True, init_center=init_center)

    # ---------- Define deep learning + TSK -------------
    model = nn.Sequential(
        NeuralNetworks(),
        TSK(in_dim=X.shape[1], out_dim=n_class, n_rule=n_rule, antecedent=gmf, order=order),
    )


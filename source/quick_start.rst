Quick Start
=================================

Training with gradient descent
##############################

Complete code can be found at: https://github.com/YuqiCui/PyTSK/quickstart_gradient_descent.py

Import everything you need::

    import numpy as np
    import torch
    import torch.nn as nn
    from pytsk.gradient_descent.antecedent import AntecedentGMF, AntecedentShareGMF, antecedent_init_center
    from pytsk.gradient_descent.callbacks import EarlyStoppingACC
    from pytsk.gradient_descent.training import Wrapper
    from pytsk.gradient_descent.tsk import TSK
    from pmlb import fetch_data
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from torch.optim import AdamW

Prepare dataset::

    # Prepare dataset by the PMLB package
    X, y = fetch_data('segmentation', return_X_y=True, local_cache_dir='./data/')
    n_class = len(np.unique(y))  # Num. of class

    # split train-test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print("Train on {} samples, test on {} samples, num. of features is {}, num. of class is {}".format(
        x_train.shape[0], x_test.shape[0], x_train.shape[1], n_class
    ))

    # Z-score
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)

Define TSK parameters::

    # Define TSK model parameters
    n_rule = 30  # Num. of rules
    lr = 0.01  # learning rate
    weight_decay = 1e-8
    consbn = False
    order = 1

Construct TSK model, for example, HTSK model with LN-ReLU::

   # --------- Define antecedent ------------
    init_center = antecedent_init_center(x_train, y_train, n_rule=n_rule)
    gmf = nn.Sequential(
        AntecedentGMF(in_dim=X.shape[1], n_rule=n_rule, high_dim=True, init_center=init_center),
        nn.LayerNorm(n_rule),
        nn.ReLU()
    )
    # set high_dim=True is highly recommended.
        nn.Dropout(p=0.25)
    )

    # --------- Define full TSK model ------------
    model = TSK(in_dim=X.shape[1], out_dim=n_class, n_rule=n_rule, antecedent=gmf, order=order, precons=None)

Define optimizer, split train-val, define earlystopping callback::

    # ----------------- optimizer ----------------------------
    ante_param, other_param = [], []
    for n, p in model.named_parameters():
        if "center" in n or "sigma" in n:
            ante_param.append(p)
        else:
            other_param.append(p)
    optimizer = AdamW(
        [{'params': ante_param, "weight_decay": 0},
        {'params': other_param, "weight_decay": weight_decay},],
        lr=lr
    )
    # ----------------- split 10% data for earlystopping -----------------
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
    # ----------------- define the earlystopping callback -----------------
    EACC = EarlyStoppingACC(x_val, y_val, verbose=1, patience=20, save_path="tmp.pkl")

Train TSK model::

    wrapper = Wrapper(model, optimizer=optimizer, criterion=nn.CrossEntropyLoss(),
                  epochs=300, callbacks=[EACC])
    wrapper.fit(x_train, y_train)
    wrapper.load("tmp.pkl")

Evaluate model's performance::

    y_pred = wrapper.predict(x_test).argmax(axis=1)
    print("[TSK] ACC: {:.4f}".format(accuracy_score(y_test, y_pred)))


Training with fuzzy clustering
###############################

Complete code can be found at: https://github.com/YuqiCui/PyTSK/quickstart_fuzzy_clustering.py

Import everything you need::

    import numpy as np
    from pmlb import fetch_data
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from pytsk.cluster.cluster import FuzzyCMeans
    from sklearn.pipeline import Pipeline

Prepare dataset::

    # Prepare dataset by the PMLB package
    X, y = fetch_data('segmentation', return_X_y=True, local_cache_dir='./data/')
    n_class = len(np.unique(y))  # Num. of class

    # split train-test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print("Train on {} samples, test on {} samples, num. of features is {}, num. of class is {}".format(
        x_train.shape[0], x_test.shape[0], x_train.shape[1], n_class
    ))

    # Z-score
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)

Define & train the TSK model::

    # --------------- Fit and predict ---------------
    n_rule = 20
    model = Pipeline(
        steps=[
            ("Antecedent", FuzzyCMeans(n_rule, sigma_scale="auto", fuzzy_index="auto")),
            ("Consequent", RidgeClassifier())
        ]
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("ACC: {:.4f}".format(accuracy_score(y_test, y_pred)))

If you need analysis the input of consequent part::

    # ---------------- get the input of consequent part for further analysis-----------------
    antecedent = model.named_steps['GaussianAntecedent']
    consequent_input = model.transform(x_test)

If you need grid search all important parameters::

        param_grid = {
        "Consequent__alpha": [0.01, 0.1, 1, 10, 100],
        "GaussianAntecedent__n_rule": [10, 20, 30, 40],
        "GaussianAntecedent__sigma_scale": [0.01, 0.1, 1, 10, 100],
        "GaussianAntecedent__fuzzy_index": ["auto", 1.8, 2, 2.2],
    }
    search = GridSearchCV(model, param_grid, n_jobs=2, cv=5, verbose=10)
    search.fit(x_train, y_train)
    y_pred = search.predict(x_test)
    print("ACC: {:.4f}".format(accuracy_score(y_test, y_pred)))

Evaluate model's performance::

    y_pred = wrapper.predict(x_test).argmax(axis=1)
    print("[TSK] ACC: {:.4f}".format(accuracy_score(y_test, y_pred)))

Complete code can be found at: https://github.com/YuqiCui/PyTSK/quick_start.py

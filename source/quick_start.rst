Quick Start
=================================

Training with gradient descent
##############################

Import everything you need::

   import numpy as np
   from torch.optim import Adam
   from torch.nn import CrossEntropyLoss
   import pytsk.torch_model as TM

Define and initialize a model::

   model = TM.TSK(in_dim, out_dim, n_rules, "tsk")
   model.init_model(x_train)

Define the optimizer, loss function, callbacks and Trainer::

   optimizer = Adam(model.parameters(), lr=0.01)
   # recommended optimizer is Adam and its variations
   # recommended learning rate is 0.01

   criterion = CrossEntropyLoss()
   es = TM.EarlyStopping(val_loader, metrics="acc", patience=20,
      larger_is_better=True, eps=1e-4, save_path="model.pkl",
      only_save_best=True)
   # train model with early-stopping

   cp = TM.CheckPerformance(test_loader, metrics="acc", name="Test")
   # monitor test accuracy during the training

   T = TM.Trainer(model, optimizer, criterion, device="cuda", callbacks=[es, cp], verbose=1)

Train the model::

   T.fit(x_train, y_train, max_epoch=100, batch_size=32)

Load the best model saved by early-stopping::

   model.load("model.pkl")

Predict the probability of test samples::

   y_pred = model.predict_score(x_test)

Evaluate model's performance::

   y_pred = np.argmax(y_pred, axis=1)
   acc = np.mean(y_pred == y_test)
   print("Test ACC: {:.4f}".format(acc))

Complete code can be found at: https://github.com/YuqiCui/PyTSK
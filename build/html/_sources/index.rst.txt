.. PyTSK documentation master file, created by
   sphinx-quickstart on Mon Jan 11 14:48:13 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyTSK's documentation!
=================================
PyTSK is a package for conveniently developing a TSK-type fuzzy neural networks.
It's dependencies are as follows:

* `Scikit-learn <https://scikit-learn.org/>`_ [Necessary] for machine learning operations.
* `Numpy <https://numpy.org/>`_ [Necessary] for matrix computing operations.
* `Scipy <https://scipy.org/>`_ [Necessary] for matrix computing operations.
* `PyTorch <https://pytorch.org>`_ [Necessary] for constructing and training fuzzy neural networks.
* `Faiss <https://github.com/facebookresearch/faiss>`_ [Optional] a faster version for k-means clustering.

To run the code in `quick start <quick_start.html>`_, you also need to install the following packages:

* `PMLB <https://epistasislab.github.io/pmlb/>`_ for downloading datasets.



.. toctree::
   :caption: Table of Contents
   :glob:
   :maxdepth: 2

   install
   quick_start
   models
   apis/*
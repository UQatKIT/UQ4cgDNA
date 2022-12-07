Getting Started
===============

uq4cgDNA is a small Python package for uncertainty quantification in the 
`cgDNA <https://lcvmwww.epfl.ch/research/cgDNA/>`_  framework. cgDNA (*coarse-grained DNA*)
denotes a simplified approach to the modeling of the mechanical properties of DNA sequences,
such as shape and stiffness. The underlying theory relies on the interpretation of DNA as
a bichain of rigid bodies. Such a model can be characterized by a reduced set of
parameters that can be inferred from molecular dynamics simulations.

The goal of this software package is the quantification of the robustness of the cgDNA model.
It thereby relies on a ' `Bayesian <https://en.wikipedia.org/wiki/Bayesian_inference>`_
problem formulation for the deduction of a probability distribution and corresponding 
metrics of the model parametrization given the MD data. Due to the complex internals of the
approach, such a distribution function is not available in closed form. Thus uq4cgDNA
includes a sampling procedure via the  `Metropolis-Hastings algorithm
<https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm>`_
from the class of *Markov-Chain Monte Carlo* sampling algorithms. Due to the high
dimensionality of the considered variables, the MH algorithm utilizes dimension-independent
proposal generators (currently the  `Preconditioned Crank-Nicholson Random Walk
<https://en.wikipedia.org/wiki/Preconditioned_Crank%E2%80%93Nicolson_algorithm>`_).

-----------------------------

Requirements and Installation
-----------------------------

uq4cgDNA is designed with a minimal collection of computational tools. Next to Python itself,
the package solely relies on the established `Numpy <https://numpy.org/>`_ and
`Scipy <https://www.scipy.org/>`_ libraries.

.. admonition:: The implementation has been tested with the following configurations

   * Windows 10, Python 3.7.7, Numpy 1.18.5, Scipy 1.5.0
   * Ubuntu 20.04, Python 3.8.2, Numpy 1.18.4, Scipy 1.4.1

For execution on windows, it is recommended to use a virtual environment and package manager
like `Anaconda <https://www.anaconda.com/products/individual>`_. In the case of Ubuntu,
the necessary packages are already integrated into the initial setup of the operating system.

.. tip::
    The `Sphinx <https://www.sphinx-doc.org/en/master/>`_ documentation tools
    (version 3.1.1) has been utilized to generate this documentation. Additionally,
    the `Read the Docs <https://www.sphinx-doc.org/en/master/>`_ theme has been used.
    
-----------------------------
    
Structure of the Repository
-----------------------------

``data:`` MD and cgDNA data to be used for computation, txt format

``doc:`` General and class documentation files

``test:`` Unittest implementations with own data

``uq4cgDNA:`` Main package

``README.md:`` Introductory file containing main content of the documentation

``main.py``: Examplary application of the package
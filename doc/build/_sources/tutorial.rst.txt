Tutorial
========

The following section serves as a short introduction to the workflow of the uq4cgDNA
package. For more detailed elaborations on the underlying implementation, please
refer to the :doc:`api` documentation. The presented code corresponds to the `main.py`
file, which can simple be executed by a suitable Python editor.

--------

Import
------

uq4cgDNA relies on modular construction principles, sectioning the implementation into
five main components:

``IOHandler:`` This module provides all functionalities for input and output from and to
files and to the console. It is therefore widely utilized by the other modules.

``cgDNAModel:`` The connection between the generic Bayesian inferrence/sampling procedure
and cgDNA-specific internals is implemented here.

``Proposal:`` This module realizes a generic Preconditioned Crank-Nicholson Random Walk 
proposal function. Future work will extend this module to other schemes.

``Posterior:`` The posterior distribution implemented in this module generates the output 
samples from the Bayesian inferrence procedure

``MCMCSampler:`` The MCMC module contains an implementation of the Metropolis-Hastings 
algorithm relying on the previous modules. 

These modules need to be imported from the package. Please note that the uq4cgDNA path 
generally needs to be known to the interpreter or has to be specified within the import 
statements:

.. code-block:: python

   from uq4cgDNA import io_handler
   from uq4cgDNA import cgdna_model
   from uq4cgDNA import proposal
   from uq4cgDNA import posterior
   from uq4cgDNA import mcmc

--------

Settings
--------

uq4cgDNA offers a variety of settings that can be utilized to customize the computation.
These settings can mainly be subdivided into three groups:

Input and Output file Settings:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   
These settings control the files that data is read from and written to, respectively. 
Additionally, they contain options regarding printing and logging. Specifically, the 
following variables need to be specified within a dictionary:

1. *Molecular dynamics data file names:* Files containing DNA Sequences along with their 
   shape and stiffness arrays
2. *cgDNA data file names:* Files containing the cgDNA parameter set encoding and 
   shape/stiffness parameter arrays for the different monomers and dimers
3. *Matrix file names:* Files containing the covariance matrices for prior and proposal 
   distributions
4. *Output and logging file names:* Destination for output samples and logging during 
   computation. Additionally, flags for logging and output to screen can be set

.. code-block:: python

   fileSettings = {
        # MD Data
        "File_MD_Sequences":                "data/Sequences.txt",
        "File_MD_Shapes":                   "data/Shapes_MD.txt",
        "File_MD_Stiffnesses":              "data/Stiffness_MD.txt",
        # cgDNA Data
        "File_Encoding":                    "data/Encoding.txt",
        "File_cgDNA_OneMerSig":             "data/OneMerSig.txt",
        "File_cgDNA_OneMerStiffness":       "data/OneMerStiff.txt",
        "File_cgDNA_TwoMerSig":             "data/TwoMerSig.txt",
        "File_cgDNA_TwoMerStiffness":       "data/TwoMerStiff.txt",
        "File_Proposal_OneMerSig":          "data/OneMerSig.txt",
        "File_Proposal_OneMerStiffness":    "data/OneMerStiff.txt",
        "File_Proposal_TwoMerSig":          "data/TwoMerSig.txt",
        "File_Proposal_TwoMerStiffness":    "data/TwoMerStiff.txt",
        # Hessian Matrix
        "File_Prior_Matrix":                "data/Hessian.txt",
        "File_Proposal_Matrix":             "data/Hessian.txt",
        # Output
        "File_Logs":                        "log.txt",
        "File_Output":                      "Output.txt",
        "Logging":                          True,
        "Printing":                         True
        }

MCMC Sampler Settings:
^^^^^^^^^^^^^^^^^^^^^^
    
These settings allow control over the Sampler itself. Again, the necessary variables
need to be specified within a dictionary:
    
1. Number of samples to be computed as usable output
2. Number of burn-in samples (these samples will be disregarded)
3. Mean batch size: Size of sample batches whose mean values (more precisely their
   difference) can be used as a convergence check
4. Statistics computation: Number of steps until statistics are re-computed
5. Output: Number of steps until output is printed to screen/log file

.. code-block:: python

   mcmcSettings = {
            "Number_of_Samples":                3,
            "Burn_in_Period":                   0,
            "Mean_Batch_Size":                  5,
            "Statistics_Interval":              10,
            "Output_Interval":                  10,
            }

Proposal Settings:
^^^^^^^^^^^^^^^^^^

For the current implementation of the pCN proposal, only one settings has to be specified,
namely the parameter determining the asymmetry of the random walk.

.. code-block:: python

   cnParameter = 1e-3

--------

Initialization
--------------

Given these settings, it is quite straight forward to start a computation. Firstly, 
the corresponding objects of the imported modules need to be initialized.
The ``IOHandler`` object is constructed using the file settings.

.. code-block:: python

   IOHandler = io_handler.IOHandler(fileSettings)

The ``cgDNAModel`` class is invoked using the ``IOHandler`` object to load the necessary 
cgDNA data. Additionally, the mode *cgdna* clarifies that the cgDNA parameter set 
(not the proposal set) is used for initialization. The ``cgDNAModel`` object can also 
be initialized by a reduced parameter set vector.

.. code-block:: python

   cgDNAModel = cgdna_model.cgDNAModel(IOHandler, "cgdna")  

In contrast to the previous objects, the ``Proposal`` is generated by a factory method, 
which allows greater flexibility regarding different initialization strategies. 
The arguments for an initialization from file are a valid ``IOHandler`` object and 
the pCN parameter. Alternatively, the proposal can be initialized with identity 
covariance matrix.

.. code-block:: python

   Proposal = proposal.pCNProposal.from_file_cov(IOHandler, cnParameter) 

The ``Posterior`` object is constructed from valid ``IOHandler`` and ``cgDNAModel`` 
objects.

.. code-block:: python
   
   Posterior = posterior.Posterior(IOHandler, cgDNAModel) 

Lastly, the ``MCMCSampler`` is generated with the corresponding settings, along with
``cgDNAModel`` and ``Posterior`` objects that provide the initial candidate and
corresponding probability values. Note that the cgDNA model is updated to contain the
initial candidate set.

.. code-block:: python

   cgDNAModel.update_from_file(IOHandler, "proposal")
   Sampler = mcmc.MCMCSampler(mcmcSettings, cgDNAModel, Posterior)

--------

Run
---
The sampler is invoked via a single command. Depending on the output flags, the resulting
samples are stored in the output file and information is printed to the screen and the 
specified log file.

.. code-block:: python

   Sampler.run(IOHandler, cgDNAModel, Proposal, Posterior)
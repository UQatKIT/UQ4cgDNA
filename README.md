![UQ Chair logo](doc/logo_uq.svg)

# **UQ4cgDNA-Python**

uq4cgDNA is a small Python package for uncertainty quantification in the [cgDNA](https://lcvmwww.epfl.ch/research/cgDNA/)  framework. cgDNA (_coarse-grained DNA_) denotes a simplified approach to the modeling of the mechanical properties of DNA sequences, such as shape and stiffness. The underlying theory relies on the interpretation of DNA as a bichain of rigid bodies. Such a model can be characterized by a reduced set of parameters that can be inferred from molecular dynamics simulations.

The goal of this software package is the quantification of the robustness of the cgDNA model. It thereby relies on a [Bayesian](https://en.wikipedia.org/wiki/Bayesian_inference) problem formulation for the deduction of a probability distribution and corresponding metrics of the model parametrization given the MD data. Due to the complex internals of the approach, such a distribution function is not available in closed form. Thus uq4cgDNA includes a sampling procedure via the  [Metropolis-Hastings algorithm](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm) from the class of _Markov-Chain Monte Carlo_ sampling algorithms. Due to the high dimensionality of the considered variables, the MH algorithm utilizes dimension-independent proposal generators (currently the  [Preconditioned Crank-Nicholson Random Walk](https://en.wikipedia.org/wiki/Preconditioned_Crank%E2%80%93Nicolson_algorithm)).

For more information regarding the underlying theory and the structure and capabilities of the uq4cgDNA software package, please refer to the documentation. 

<br>

## Requirements and Installation

uq4cgDNA is designed with a minimal collection of computational tools. Next to Python itself, the package solely relies on the established [Numpy](https://numpy.org/) and [Scipy](https://www.scipy.org/) libraries.

The implementation has been tested with the following configurations:
- Windows 10, Python 3.7.7, Numpy 1.18.5, Scipy 1.5.0
- Ubuntu 20.04, Python 3.8.2, Numpy 1.18.4, Scipy 1.4.1

For execution on windows, it is recommended to use a virtual environment and package manager like [Anaconda](https://www.anaconda.com/products/individual). In the case of Ubuntu, the necessary packages are already integrated into the initial setup of the operating system.

<br>

## Structure of the Repository
`data:` MD and cgDNA data to be used for computation, txt format

`doc:` General and class documentation files

`test:` Unittest implementations with own data

`uq4cgDNA:` Main package

`README.md:` This file

`main.py`: Examplary application of the package

<br>

## A Short Tutorial

The following section serves as a short introduction to the workflow of the uq4cgDNA package. For more detailed elaborations on the underlying implementation, please refer to the documentation. The presented code corresponds to the `main.py` file, which can simple be executed by a suitable Python editor.

### Import

uq4cgDNA relies on modular construction principles, sectioning the implementation into five main components:

`IOHandler:` This module provides all functionalities for input and output from and to files and to the console. It is therefore widely utilized by the other modules.

`cgDNAModel:` The connection between the generic Bayesian inferrence/sampling procedure and cgDNA-specific internals is implemented here.

`Proposal:` This module realizes a generic Preconditioned Crank-Nicholson Random Walk proposal function. Future work will extend this module to other schemes.

`Posterior:` The posterior distribution implemented in this module generates the output samples from the Bayesian inferrence procedure

`MCMCSampler:` The MCMC module contains an implementation of the Metropolis-Hastings algorithm relying on the previous modules. 

These modules need to be imported from the package. Please note that the uq4cgDNA path generally needs to be known to the interpreter or has to be specified within the import statements:
```python
from uq4cgDNA import io_handler
from uq4cgDNA import cgdna_model
from uq4cgDNA import proposal
from uq4cgDNA import posterior
from uq4cgDNA import mcmc
```

### Settings

uq4cgDNA offers a variety of settings that can be utilized to customize the computation. These settings can mainly be subdivided into three groups:

**Input and Output file Settings:**
   
These settings control the files that data is read from and written to, respectively. Additionally, they contain options regarding printing and logging. Specifically, the following variables need to be specified within a dictionary:

1. Molecular dynamics data file names: Files containing DNA Sequences along with their shape and stiffness arrays
2. cgDNA data file names: Files containing the cgDNA parameter set encoding and shape/stiffness parameter arrays for the different monomers and dimers
3. Matrix file names: Files containing the covariance matrices for prior and proposal distributions
4. Output and logging file names: Destination for output samples and logging during computation. Additionally, flags for logging and output to screen can be set

```python
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
```

**MCMC Sampler Settings:**
    
These settings allow control over the Sampler itself. Again, the necessary variables need to be specified within a dictionary:
    
1. Number of samples to be computed as usable output
2. Number of burn-in samples (these samples will be disregarded)
3. Mean batch size: Size of sample batches whose mean values (more precisely their difference) can be used as a convergence check
4. Statistics computation: Number of steps until statistics are re-computed
5. Output: Number of steps until output is printed to screen/log file

```python
mcmcSettings = {
        "Number_of_Samples":                3,
        "Burn_in_Period":                   0,
        "Mean_Batch_Size":                  5,
        "Statistics_Interval":              10,
        "Output_Interval":                  10,
        }
```

**Proposal Settings:**

For the current implementation of the pCN proposal, only one settings has to be specified, namely the parameter determining the asymmetry of the random walk.

```python
cnParameter = 1e-3
```

### Initialization

Given these settings, it is quite straight forward to start a computation. Firstly, the corresponding objects of the imported modules need to be initialized.
The `IOHandler` object is constructed using the file settings.
```python
IOHandler = io_handler.IOHandler(fileSettings)
```

The `cgDNAModel class` is invoked using the `IOHandler` object to load the necessary cgDNA data. Additionally, the mode _cgdna_ clarifies that the cgDNA parameter set (not the proposal set) is used for initialization. The `cgDNAModel` object can also be initialized by a reduced parameter set vector.
```python
cgDNAModel = cgdna_model.cgDNAModel(IOHandler, "cgdna")  
```

In contrast to the previous objects, the `Proposal` is generated by a factory method, which allows greater flexibility regarding different initialization strategies. The arguments for an initialization from file are a valid `IOHandler` object and the pCN parameter. Alternatively, the proposal can be initialized with identity covariance matrix.
```python
Proposal = proposal.pCNProposal.from_file_cov(IOHandler, cnParameter) 
```

The `Posterior` object is constructed from valid `IOHandler` and `cgDNAModel` objects.
```python
Posterior = posterior.Posterior(IOHandler, cgDNAModel) 
```

Lastly, the `MCMCSampler` is generated with the corresponding settings, along with `cgDNAModel` and `Posterior` objects that provide the initial candidate and cossesponding probability values. Note that the cgDNA model is updated to contain the initial candidate set.
```python
cgDNAModel.update_from_file(IOHandler, "proposal")
Sampler = mcmc.MCMCSampler(mcmcSettings, cgDNAModel, Posterior)
```

### Run
The sampler is invoked via a single command. Depending on the output flags, the resulting samples are stored in the output file and information is printed to the screen and the specified log file.

```python
Sampler.run(IOHandler, cgDNAModel, Proposal, Posterior)
```

<br>

## Issues and To Dos

- Viable computations only for very small value of pCN parameter: Correct implementation?
- Add test module to autodoc
- Store all samples AFTER each MCMC iteration 
- If sample is not PD, compute new one within same iteration
- Check dimer encoding
- Discuss encapsulation of object members for easier initialization
- Discuss implementation of a wrapper application
- Catch exception raised when proposal probability is too low for restart
- Implement restart function from file

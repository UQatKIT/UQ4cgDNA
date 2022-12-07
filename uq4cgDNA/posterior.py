##############################################################################
#                        _  _             ____  _   _    _                   #
#            _   _  __ _| || |   ___ __ _|  _ \| \ | |  / \                  #
#           | | | |/ _` | || |_ / __/ _` | | | |  \| | / _ \                 #
#           | |_| | (_| |__   _| (_| (_| | |_| | |\  |/ ___ \                #
#            \__,_|\__, |  |_|  \___\__, |____/|_| \_/_/   \_\               #
#                     |_|           |___/                                    #
#                                                                            #
##############################################################################

""" Posterior Distribution for Bayesian estimates from uq4cgDNA

This module comprises the functionalities for the computation of posterior
probabilites in the uq4cgDNA framework. The necessary data structures are 
inferred from IOHandler and cgDNAModel objects.
The actual computation of the posterior probability in the bayesian context
consists of a likelihood term and a prior probability. The likelihood is 
computed as the Kullback-Leibler divergence of two gaussians characterized by 
MD and assembled cgDNA data structures, respectively. Since the computation
is performance-critical, the corresponding routines are optimized for a minimum
number of matrix manipulations.The prior is computed as a gaussian consisting 
of a matrix read from file and the cgDNA parameter vector.
Given a proper initialization with IOHandler and cgDNAModel objects, objects
of the posterior class automatically compute the posterior probability for 
any given candidate model. To create viable values for the sample 
probabilities, the exponent in the resulting formulation is normalized by the
cumulative Kullback-Leibler divergence of the cgDNA parameterset.

Note: The underlying distribution funtion is NOT normalized, making it useful
      only in contexts where ratios are required.


Author Sebastian Krumscheidt, Maximilian Kruse
Date 24.06.2020
"""

#=========================== Preliminary Commands ============================
import numpy as np
from scipy.sparse.linalg import splu

#============================= Posterior Class ===============================

class Posterior:
    """ Posterior class for the uq4cgDNA framework.
    
    This class provides the functionalities for the Posterior module. Objects
    of this class can be utilized to generate samples in an MCMC procedure.
    
    Attributes
    ----------
    MD sequence list, shapes and stifnesses : 
        Lists of strings and 1/2D numpy float arrays
    Prior matrix (inverse) and vector : 1/2D numpy float arrays
    MD matrix determinants : List of floats
    Normalization factor: float
    
    Methods
    -------
    __init__(IOHandler, cgDNAModel)
        Sets up object structures, computes prior inverse and MD determinants,
        compute normalization factor
    __compute_normalization_constant(candidateModel)
        Computes divergence for cgdna parameterset to use for normalization   
    __compute_kl_divergence(cgDNADecomposition, cgDNAVector,\
                            MDMatrix, MDVector, MDDeterminant)    
        Computes divergence for given candidate model and DNA sequence
    compute_posterior_probability(candidateModel)
        For given model, computes posterior probability from sum of divergences
        over all sequences and prior distribution
    get_prior_data()
        Returns covariance matrix and mean vector from prior distribution
    get_md_determinants()
        Returns determinants of MD stiffness matrices
    get_normalization_factor()
        Returns normalization constant from Kullback-Leibler divergence with 
        respect to the cgDNA parameter set
    """

    #-------------------------------------------------------------------------
    def __init__(self, IOHandler, cgDNAModel): 
        """ Constructor
        
        The constructor initializes all the data structures necessary for the 
        computation of the Kullback Leibler divergence and the prior term.
        Note that the cgDNA model utilized as input has to contain the
        parameter set obtained from the cgDNA optimization procedure.
        
        Parameters
        ----------
        IOHandler : IOHandler class object
        cgDNAModel : cgDNAModel class object 
            Holds optimal parameter set
        """
        
        # Read in MD Data and Hessian
        self.__sequenceList, self.__shapeVectors, self.__stiffnessMatrices \
            = IOHandler.read_md_data()
        self.__priorMatrix = IOHandler.read_prior_matrix()
        self.__priorVector = cgDNAModel.get_parameter_vector()
        # Invert Hessian for prior matrix
        self.__priorMatrix = np.linalg.inv(self.__priorMatrix)
        # Compute determinants of all MD matrices
        self.__MDDeterminants = [np.abs(np.prod(splu(matrix).U.diagonal())) 
                                 for matrix in self.__stiffnessMatrices]
        # Compute normalization factor for divergence term
        self.__normFactor = self.__compute_normalization_constant(cgDNAModel)
        
    #-------------------------------------------------------------------------
    def __compute_normalization_constant(self, candidateModel):
        """ Computes normalization constant for posterior
        
        To generate viable posterior probabilites, the exponential of the 
        probability function is normalized by the accumulated divergence from
        the optimal cgDNA parameter set.
        
        Parameters
        ----------
        candidateModel : cgDNAModel class object 
            Holds optimal parameter set

        Returns
        -------
        normFactor : float
            Normalization factor for kl divergence
        """
        
        normFactor = 0      
        
        for i in range(len(self.__sequenceList)):
            sequence = self.__sequenceList[i]
            stiffnessMatrix, sigmaVector =\
                candidateModel.assemble_model(sequence)
            cgDNADecomposition = splu(stiffnessMatrix)
            stiffnessMatrix = stiffnessMatrix.todense()
            cgDNAVector = cgDNADecomposition.solve(sigmaVector)
            normFactor +=\
                self.__compute_kl_divergence(cgDNADecomposition,
                                             cgDNAVector,                                                   
                                             self.__stiffnessMatrices[i],
                                             self.__shapeVectors[i],
                                             self.__MDDeterminants[i])
                    
        return normFactor

    #-------------------------------------------------------------------------
    def __compute_kl_divergence(self, cgDNADecomposition, cgDNAVector,
                                MDMatrix, MDVector, MDDeterminant):
        """ Computes Kullback Leibler divergence for given model and sequence
        
        The computation of the Kullback Leibler divergence is the most 
        computationally expensive operation in the uq4cgDNA framework, since
        it is called for every parameter set and every MD sequence. The 
        computation mainly relies on precomputed quantities to optimize its
        efficiency.
        
        Parameters
        ----------
        cgDNADecomposition : Scipy SuperLU object
            Sparse LU-decomposition of the global cgDNA stiffness matrix
        cgDNAVector : 1D numpy float array
            global cgDNA shape vector
        MDMatrix : Scipy csc matrix object
            Sparse representation of MD matrix
        MDVector : 1D numpy float array
            MD Vector
        MDDeterminant : float
            Determinant of the current MD matrix

        Returns
        -------
        Kullback Leibler divergence: float
        """
        
        assert cgDNADecomposition.shape == MDMatrix.shape,\
            "cgDNA and MD Matrix do not have the same shape" 
        assert cgDNAVector.shape == MDVector.shape,\
            "cgDNA and MD Vector do not have the same shape"
       
        traceTerm = np.sum(np.diagonal(cgDNADecomposition.solve(
                    MDMatrix.todense())))
        detTerm = np.log(np.abs(np.prod(cgDNADecomposition.U.diagonal())) 
                / MDDeterminant)
        vecTerm = (cgDNAVector-MDVector) @ MDMatrix @ (cgDNAVector-MDVector)
        
        return 0.5 * (traceTerm + detTerm + vecTerm - len(cgDNAVector))
    
    #-------------------------------------------------------------------------
    def compute_posterior_probability(self, candidateModel):
        """ Computes Posterior probability
        
        The posterior probability is formed from a likelihood function
        generated from the exponential of the acccumulated KL-divergence over
        all sequences and a Gaussian prior. The exponential is normalized
        (see '__compute_normalization_constant'). Note that the overall pdf is 
        NOT normalized.
        
        Parameters
        ----------
        candidateModel : cgDNAModel class object 
            Current candidate model to compute divergence for

        Returns
        -------
        Posterior probability: float
        """
        
        KLDivergence = 0      
        
        for i in range(len(self.__sequenceList)):
            sequence = self.__sequenceList[i]
            stiffnessMatrix, sigmaVector =\
                candidateModel.assemble_model(sequence)
            cgDNADecomposition = splu(stiffnessMatrix)
            stiffnessMatrix = stiffnessMatrix.todense()
            cgDNAVector = cgDNADecomposition.solve(sigmaVector)
            KLDivergence +=\
                self.__compute_kl_divergence(cgDNADecomposition,
                                             cgDNAVector,                                                   
                                             self.__stiffnessMatrices[i],
                                             self.__shapeVectors[i],
                                             self.__MDDeterminants[i])
                    
        candidateVector = candidateModel.get_parameter_vector()
        assert candidateVector.shape == self.__priorVector.shape,\
            "Candidate and prior vector have different shapes"
        
        vecDiff = candidateVector - self.__priorVector
        priorVal = 0.5 * vecDiff @ self.__priorMatrix @ vecDiff
        
        return np.exp(-(KLDivergence + priorVal) / self.__normFactor)
    
    #-------------------------------------------------------------------------
    def get_prior_data(self):
        """ Returns covariance matrix and mean vector of prior distribution
        
        Returns
        -------
        2D float array
            Prior covariance matrix
        1D float array
            Prior mean vector
        """
        
        return self.__priorMatrix, self.__priorVector
    
    #-------------------------------------------------------------------------
    def get_md_determinants(self):
        """ Returns determinants of MD stiffness matrices
        
        Returns
        -------
        List of floats
            Determinant values
        """
        
        return self.__MDDeterminants
    
    #-------------------------------------------------------------------------
    def get_normalization_factor(self):
        """ Returns normalization factor for Kullback-Leibler divergence
        
        To obtain feasible values, the Kullback-Leibler divergence is
        normalized by its value with respect to the cgDNA parameter set
        (also see above)

        Returns
        -------
        float
            Value of normalization constant
        """
        
        return self.__normFactor
##############################################################################
#                        _  _             ____  _   _    _                   #
#            _   _  __ _| || |   ___ __ _|  _ \| \ | |  / \                  #
#           | | | |/ _` | || |_ / __/ _` | | | |  \| | / _ \                 #
#           | |_| | (_| |__   _| (_| (_| | |_| | |\  |/ ___ \                #
#            \__,_|\__, |  |_|  \___\__, |____/|_| \_/_/   \_\               #
#                     |_|           |___/                                    #
#                                                                            #
##############################################################################

""" Proposal Generator for uq4cgDNA

This module contains proposal distributions for the cgDNA model. Objects of
the below classes can generate candidates of cgDNA parameter sets for an MCMC
algorithm. The model is planned to contain a variety of proposal algorithms
for different use cases. The proposal has a two-fold function within an MCMC
algorithm. Firstly it generates new sample candidates given a current one. 
Secondly, it assess their probability via a conditional pdf w.r.t to the last
candidate.

Currently implemented:
    - Preconditioned Crank-Nicolson Randowm Walk


Author Sebastian Krumscheidt, Maximilian Kruse
Date 09.07.2020
"""

#=========================== Preliminary Commands ============================
import numpy as np
import scipy as sp

#============================ Proposal Classes ===============================
  
class pCNProposal():
    """ Proposal class implementing preconditioned Crank-Nicolson Random Walk
    
    This class generates pCN proposals. It can be initialized with a default
    identity covariance matrix or a matrix read from an external file. These
    construction alternatives are realized via factory methods. Moreover, the
    class provides the functionality for computing conditional probabilities.
    
    The pCN procedure creates samples as:
        
        X_n+1 = sqrt(1-beta^2)*X_n + beta*Xi
        
    With beta as tuning parameter and Xi ~ N(0,covMatrix)
    
    Attributes
    ----------
    covMatrix : 2D numpy float array or numpy sparse csc matrix
        Covariance matrix for the generation of gaussian random vectors
    invCovMatrix: 2D numpy float array or numpy sparse csc matrix
        Inverse covariance matrix for the efficient computation of conditional
        probabilities
    cnParameter: float
        Tuning parameter of the algorithm in range (0,1)
    
    Methods
    -------
    __init__(covMatrix, invCovMatrix, cnParameter)
        Construtor (to be called by factory methods)
    from_identity_cov(cnParameter)
        Factory method for creation with identity covariance matrix
    from_file_cov(cnParameter, IOHandler)
        Factory method for creation with custom covariance matrix
    generate(candidate)
        Generates new candidate for MCMC
    compute_conditional_probability(targetVector, givenVector)
        Computes conditional probability for candidate comparison in MCMC
    get_covariance_matrices()
        Reutrns covariance matrix and inverse
    """
    
    #-------------------------------------------------------------------------
    def __init__(self, covMatrix, invCovMatrix, cnParameter):
        """ Constructor
        
        Sets the (inverse) covariance and tuning parameter internally.

        Parameters
        ----------
        covMatrix : numpy array or sparse matrix
            Covariance matrix for the generation of gaussian random vectors
        invCovMatrix : numpy array or sparse matrix
            Inverse covariance matrix for the efficient computation of 
            conditional probabilities
        cnParameter : float
            Tuning parameter of the algorithm in range (0,1)

        Raises
        ------
        ValueError
            Checks if tuning parameter is in proper range
        """
        
        if covMatrix.shape != (1944,1944):
            raise ValueError("Covariance Matrix has wrong size")
        if invCovMatrix.shape != (1944,1944):
            raise ValueError("Inverse Covariance Matrix has wrong size")
        if cnParameter >=1 or cnParameter <= 0:
            raise ValueError("cnParameter must be in interval (0,1)")
            
        self.__covMatrix = covMatrix
        self.__invCovMatrix = invCovMatrix
        self.__cnParameter = cnParameter
    
    #-------------------------------------------------------------------------
    @classmethod
    def from_identity_cov(cls, cnParameter):
        """ Factory Method for generation of porposal with default covariance
            equal to the identity matrix.
        
        Parameters
        ----------
        cnParameter : float
            Tuning parameter of the algorithm in range (0,1)

        Returns
        -------
        pCNProposal
            pCNProposal with identity covariance matrix
        """
        
        covMatrix = sp.sparse.identity(1944)
        invCovMatrix = covMatrix
        
        return cls(covMatrix, invCovMatrix, cnParameter)

    #-------------------------------------------------------------------------
    @classmethod    
    def from_file_cov(cls, IOHandler, cnParameter):
        """ Factory Method for generation of porposal with custom covariance
        
        The generation of multivariate Gaussians relies on the preliminary
        computation of the cholesky decomposition of the covariance matrix.
        
        Parameters
        ----------
        IOHandler
            IOHandler object for file reading
        cnParameter : float
            Tuning parameter of the algorithm in range (0,1)

        Returns
        -------
        pCNProposal
            pCNProposal with custom covariance matrix
        """
        
        covMatrix = IOHandler.read_proposal_matrix()
        invCovMatrix = np.linalg.inv(covMatrix)
        covMatrix = np.linalg.cholesky(covMatrix)
        
        return cls(covMatrix, invCovMatrix, cnParameter)
    
    #-------------------------------------------------------------------------
    def generate(self, candidate):
        """ Generates new parameter set candidate from pCN random walk 
            procedure accoring to the relation stated above
        
        Parameters
        ----------
        candidate : 1D numpy float array
            Previous candidate vector

        Returns
        -------
        newCandidate : 1D numpy float array
            Next candidate vector
        """
        
        assert len(candidate) == 1944, "Candidate vector has wrong size"
        newCandidate = np.sqrt(1 - self.__cnParameter**2) * candidate \
                     + self.__cnParameter * self.__covMatrix \
                     @ np.random.standard_normal(1944)
                     
        return newCandidate
    
    #-------------------------------------------------------------------------
    def compute_conditional_probability(self, targetVector, givenVector):
        """ Compute conitional probability of targetVector given givenVector
        
        The conditional distribution describes a Gaussian of fixed variance
        cnParameter * covMatrix and a mean value given by 
        sqrt(1-cnParameter^2)*givenVector. Note that the resulting value is 
        NOT normalized.

        Parameters
        ----------
        targetVector : 1D numpy float array
            Vector to compute probability for
        givenVector : 1D numpy float array
            Vector on which probability depends

        Returns
        -------
        condProb : float
            Conditional probability
            
        Raises
        ------
        ValueError
            Checks if calculated probability is too low to ensure viable
            further computations
        """
        
        assert len(targetVector) == 1944, "Target vector has wrong size"
        assert len(givenVector) == 1944, "Given vector has wrong size"
        vecDiff = \
            targetVector - np.sqrt(1 - self.__cnParameter**2) * givenVector
        condProb = np.exp(-0.5 / self.__cnParameter * vecDiff 
                          @ self.__invCovMatrix @ vecDiff)
        
        if condProb < 1e-10:
            raise ValueError("Conditional probability is too low, "
                             "decrease Crank-Nicolson parameter.")
            
        return condProb
    
    #-------------------------------------------------------------------------
    def get_covariance_matrices(self):
        """ Gets covariance matrix (Cholesky factorized) and inverse
        
        Returns
        -------
        2D numpy float array or Scipy sparse csc matrix
            Proposal covariance matrix (might be Cholesky decomposed)
        2D numpy float array or Scipy sparse csc matrix
            Inverse of proposal covariance matrix
        """
        
        return self.__covMatrix, self.__invCovMatrix
        
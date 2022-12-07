##############################################################################
#                        _  _             ____  _   _    _                   #
#            _   _  __ _| || |   ___ __ _|  _ \| \ | |  / \                  #
#           | | | |/ _` | || |_ / __/ _` | | | |  \| | / _ \                 #
#           | |_| | (_| |__   _| (_| (_| | |_| | |\  |/ ___ \                #
#            \__,_|\__, |  |_|  \___\__, |____/|_| \_/_/   \_\               #
#                     |_|           |___/                                    #
#                                                                            #
##############################################################################

""" MCMC Sampler for uq4cgDNA

This module provides a Markov Chain Monte Carlo algorithm for the uq4cgDNA 
framework. More precisely, it performs a Metropolis-Hastings algorithm for the
generation of samples from the Bayesian inference model.
It relies on all classes within the framework and constitutes their combination 
to create samples for the Bayesian cgDNA UQ framework. The MCMCSampler class 
takes various settings for customization, including:
    - Burn-in period
    - Mean batch convergence check
    - Acceptance statistics
    

Author Sebastian Krumscheidt, Maximilian Kruse
Date 09.07.2020
"""

#=========================== Preliminary Commands ============================
import time
import numpy as np

#============================ MCMCSampler Class ==============================

class MCMCSampler:
    """ MCMCSampler class for the uq4cgDNA framework.
    
    This class provides the functionalities for the mcmc module. Its core is 
    a Metropolis-Hastings algorithm performed until the desired number of 
    samples is generated. The necessary probabilities are computed by the other
    classes of the framework.
    
    Attributes
    ----------
    Settings for computation of statistics : list of integers
    Old and new candidate vectors : 1D numpy float arrays
    Old and new posterior probabilities : floats
    Sample counters : integers
    mean batch difference : float
    mean batch accumulation vectors : 1D numpy float arrays
    
    Methods
    -------
    __init__(settings, cgDNAModel, Posterior)
        Reads in settings, sets up initial candidate
    __compute_acceptance_probability(cgDNAModel, Proposal, Posterior)
        Computes acceptance probability of new candidate
    __update()
        Updates data structures (in case candidate is accepted)
    __compute_statistics()
        Checks ratio of positive definite and accepted candidates
    __compute_batch_diff()
        Computes the difference between consecutive batch means
    __print_output()
        Prints information to screen and log file
    run()
        Metropolis-Hastings algorithm
    get_status()
        Returns current candidate and posterior probability
    """
    
    #-------------------------------------------------------------------------
    def __init__(self, settings, cgDNAModel, Posterior):
        """ Constructor

        Parameters
        ----------
        settings : dict
            User settings for MCMC sampler
        cgDNAModel : cgDNAModel class
            Utilized model to sample from
        Posterior : Posterior class
            Object for computation of initial target distribution value

        Raises
        ------
        KeyError
            Checks validity of user input
        """
        
        # Check and assign settings
        try:
            self.__numBurnIn = settings["Burn_in_Period"]
            self.__numSamples = settings["Number_of_Samples"]       
            self.__meanbatchSize = settings["Mean_Batch_Size"]
            self.__statsInterval = settings["Statistics_Interval"]
            self.__outputInterval = settings["Output_Interval"]
        except KeyError as ke:
            raise KeyError("Setting " + str(ke) + " not correctly defined.")
        
        # Set initial candidate and its posterior probability 
        self.__lastCandidate = cgDNAModel.get_parameter_vector()
        self.__newCandidate = np.zeros(1944)
        self.__lastProbability \
            = Posterior.compute_posterior_probability(cgDNAModel)
        self.__newProbability = 0
    
    #-------------------------------------------------------------------------
    def __compute_acceptance_probability(self, cgDNAModel, 
                                         Proposal, Posterior):
        """ Computes Metropolis-Hastings acceptance probability
        
        The acceptance probability is computed from the ratios of proposal and
        posterior probabilities for the current and proposed candidate. The 
        computation via ratios eliminates the necessities for the (impossible)
        computation of normalization factors.
                
        Parameters
        ----------
        cgDNAModel : cgDNAModel class
            Utilized cgDNA model
        Proposal : Proposal class
            Proposal distribution to compute probabilities from
        Posterior : Posterior class
            Posterior distribution (target)

        Returns
        -------
        acceptanceProbability : float
            Acceptance probability for proposed candidate
        """
        
        self.__newProbability \
            = Posterior.compute_posterior_probability(cgDNAModel)
        pCondNewOld \
            = Proposal.compute_conditional_probability(self.__newCandidate,
                                                       self.__lastCandidate)
        pCondOldNew \
            = Proposal.compute_conditional_probability(self.__lastCandidate,
                                                       self.__newCandidate)
        acceptanceProbability \
            = np.minimum(1, self.__newProbability * pCondOldNew /
                        (self.__lastProbability * pCondNewOld))
            
        return acceptanceProbability
    
    #-------------------------------------------------------------------------
    def __update(self):
        """ 
        Update chain if candidate is accepted. This includes the storage of 
        the sample vector and posterior probability along with the update of
        the sampling statistics measures.        
        """
        
        self.__lastCandidate = self.__newCandidate
        self.__lastProbability = self.__newProbability
        self.__accumulatedSolution += self.__newCandidate
        self.__generatedSamples += 1
        self.__generatedSamplesBatch += 1
        self.__generatedSamplesTotal += 1
        
    #-------------------------------------------------------------------------
    def __compute_statistics(self):
        """ Compute statistics for output
        
        1. Portion of samples that are accepted
        2. Portion of samples that are positive definite
        """
        
        self.__portionPD = self.__pdSamples\
                         / self.__proposedSamples
        self.__portionGenerated = self.__generatedSamples\
                                / self.__proposedSamples
        self.__proposedSamples = 0
        self.__pdSamples = 0
        self.__generatedSamples = 0
    
    #-------------------------------------------------------------------------
    def __compute_batch_diff(self):
        """ Compute the difference in the sample mean for succeding batches.
            Batch mean difference can be used to assess convergence.
        """
        
        self.__meanBatchNew = 1 / self.__meanbatchSize\
                            * self.__accumulatedSolution
        self.__meanBatchDiff = np.linalg.norm(self.__meanBatchNew
                                            - self.__meanBatchLast)                  
        self.__meanBatchLast = np.copy(self.__meanBatchNew) 
        self.__generatedSamplesBatch = 0
        self.__accumulatedSolution.fill(0)
    
    #-------------------------------------------------------------------------
    def __print_output(self, IOHandler):
        """ Print output to screen and log file """
        
        IOHandler.log(f"{self.__iterNum:<10d}  "
                      f"{time.perf_counter()-self.__startTime:<10.3f}  "
                      f"{self.__generatedSamplesTotal:<10d}  "
                      f"{self.__portionPD:<10.3f}  "
                      f"{self.__portionGenerated:<10.3f}  "
                      f"{self.__meanBatchDiff:<10.3f}")
    
    #-------------------------------------------------------------------------    
    def run(self, IOHandler, cgDNAModel, Proposal, Posterior):
        """ Metropolis-Hastings Solver Loop
        
        Performs sampling according to MH-algorithm, invokes computation of 
        corresponding statistics, calls IOHandler for logging and output.
        
        Parameters
        ----------
        IOHandler: IOHandler class
            Utilized IOHandler class
        cgDNAModel : cgDNAModel class
            Utilized cgDNA model
        Proposal : Proposal class
            Utilized proposal distribution
        Posterior : Posterior class
            Utilized posterior distribution
        """
        
        # Set flags and counters for statistics
        self.__proposedSamples = 0
        self.__pdSamples = 0
        self.__generatedSamples = 0
        self.__portionPD = 0
        self.__portionGenerated = 0
        self.__generatedSamplesTotal = 0 
        self.__generatedSamplesBatch = 0
        self.__meanBatchDiff = 0
        self.__meanBatchLast = np.zeros(1944)
        self.__accumulatedSolution = np.zeros(1944)
        
        # Start timing and printing
        self.__startTime = time.perf_counter() 
        
        with IOHandler.start_recording(): 
            IOHandler.log(f"{'Iteration':12}{'Runtime[s]':12}{'Samples':12}"
                          f"{'PortionPD':12}{'PortionGen':12}{'BatchDiff':12}")
            
            # ------------- Sampling Loop -------------     
            self.__iterNum = 0
            while self.__generatedSamplesTotal\
            < (self.__numSamples + self.__numBurnIn):
                
                # Generate statistics and output
                if self.__generatedSamplesBatch == self.__meanbatchSize:
                    self.__compute_batch_diff()              
                if self.__iterNum % self.__statsInterval == 0 \
                and self.__iterNum != 0:
                    self.__compute_statistics()           
                if self.__iterNum % self.__outputInterval == 0 \
                and self.__iterNum != 0:
                    self.__print_output(IOHandler)
                    
                # Increment iteration counter
                self.__iterNum += 1
                
                # Create new sample
                self.__newCandidate = Proposal.generate(self.__lastCandidate)
                cgDNAModel.update_from_vector(self.__newCandidate)
                self.__proposedSamples += 1
                
                # Check if parameter set is "positive definite"
                if cgDNAModel.check_postitive_definiteness() == True:
                    self.__pdSamples += 1
                else:
                    continue
                        
                # Check if new candidate is accepted    
                uniformProb = np.random.uniform()
                acceptProb = self.__compute_acceptance_probability(cgDNAModel, 
                                                                   Proposal, 
                                                                   Posterior)
                
                # Update accepted candidates
                if uniformProb <= acceptProb:
                    self.__update()
                    if self.__generatedSamplesTotal >= self.__numBurnIn:
                        IOHandler.output(self.__newCandidate)
            
            # Write information from last iteration
            if self.__iterNum % self.__outputInterval != 0:
                self.__print_output(IOHandler)
                
    #-------------------------------------------------------------------------    
    def get_status(self):
        """ Returns current/initial status, meaning model candidate vector and
            corresponding posterior probability
        
        Returns
        -------
        1D numpy float array
            Candidate vector
        float
            Posterior probability corresponding to candidate
        """
        
        return self.__lastCandidate, self.__lastProbability 
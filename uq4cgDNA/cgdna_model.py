##############################################################################
#                        _  _             ____  _   _    _                   #
#            _   _  __ _| || |   ___ __ _|  _ \| \ | |  / \                  #
#           | | | |/ _` | || |_ / __/ _` | | | |  \| | / _ \                 #
#           | |_| | (_| |__   _| (_| (_| | |_| | |\  |/ ___ \                #
#            \__,_|\__, |  |_|  \___\__, |____/|_| \_/_/   \_\               #
#                     |_|           |___/                                    #
#                                                                            #
##############################################################################

""" cgDNA model file of the uq4cgDNA Framework

This module serves as a wrapper for cgDNA-specific tasks within the framework.
All data manipulations not part of generic linear algebra routines are
encapsulated in the cgDNAModel class. The cgDNA structures, meaning the 
shape vectors and stiffness matrices are stored in two forms:
    1. Reduced parameter vector only holding 2 independent monomers and 10 
       independent dimer pairs. Moreover, only the upper triangular part
       (including the diagonal) is stored for the symmetric stiffness matrices.
       This reduced form can be utilized for generic algebraic computations and
       is used in the cgDNA-independent part of the UQ framework
    2. Full representation of all for monomers and 16 dimer pairs. This 
       representation is used for the assembly of the cgDNA model matrix and 
       vector for a given sequence.
Note that both formulations are equivalent and can be easily converted into
each other. Particularly, the complementary monomers and dimer pairs can be
reconstructed by a simple matrix transformation.

The cgDNAModel class provides the following functionalities for the given data:
    - Inference of all data structures from given file (with the help of the
      IOHandler class). This includes the transformation between the different
      formulations
    - Checking for "positiveness" of the given parameter set. This property is
      necessary for the set to be valid
    - Assembly of the cgDNA model given a specific sequence
    
IMPORTANT: It is assumed that the encoding of the monomer and dimer blocks is
           given such that the first 10 dimers constitute and independent set
           and dimers 5-10 are complementary to 11-16.


Author Sebastian Krumscheidt, Maximilian Kruse
Date 14.06.2020
"""

#=========================== Preliminary Commands ============================
import numpy as np
from scipy.sparse import csc_matrix


#============================ cgDNAModel Class ===============================
class cgDNAModel:
    """ cgDNAModel class for the uq4cgDNA framework.
    
    This class provides the functionalities for the cgdna_model module. Objects
    of this class serve as a junction between the generic part of the 
    uncertainty quantification framework and cgDNA-specific computations
    
    Attributes
    ----------
    Dictionary for the mapping of different monomers/dimers : strings, integers
    Lists of arrays for monomer/dimer shapes and stiffnesses : 
        1/2D numpy float arrays
    Transformation matrix for complementary monomers/dimers : 
        2D numpy float array
    Reduced parameter vector : 1D numpy float array
    Supplementary matrix indices and buffer : 1/2D numpy float arrays
    
    Methods
    -------
    __init__(IOHandler)
        Sets up object structures
    __setup_model_structure(maxSequenceLength)
        Computes index arrays for cgDNA model
    __setup_transformation_matrix()
        Computes transformation matrices for complementary structures
    __complement_sequences()
        Computes complementary monomers and dimers
    update_from_file(IOHandler, mode)
        Updates structures from file (cgDNA set or other proposal)
    update_from_vector(vector)
        Updates structures from reduced parameter vector
    assemble_model(sequence)
        Builds cgDNA Model for given sequence
    get_parameter_vector()
        Returns reduced parameter vector
    get_parameter_arrays()
        Returns parameter arrays
    """
    
    #-------------------------------------------------------------------------
    def __init__(self, IOHandler, initArg):
        """ Constructor: Initializes all pre-computable structures
            - Mapping from sequence letters to arrays
            - Array lists for cgDNA structures
            - Supplementary index structures for format conversion
            - Index arrays for cgDNA model matrix and vector

        Parameters
        ----------
        IOHandler : IOHandler class object
        initArg : 1D numpy float array or string
            Depending on the type of the argument, the model is initialized
            from file or from a reduced parameter vector
        """
        
        # Dictionary for mapping from sequence letters to arrays in list
        self.__oneMerKeys, self.__twoMerKeys = IOHandler.read_encoding()
        oneMerVals = [i for i in range(4)]
        twoMerVals = [i for i in range(16)]
        self.__oneMerEncoding = dict(zip(self.__oneMerKeys, oneMerVals))
        self.__twoMerEncoding = dict(zip(self.__twoMerKeys, twoMerVals))
        
        # Data structures         
        self.__sigmaOneMer = [np.zeros(6) for i in range(4)]
        self.__sigmaTwoMer = [np.zeros(18) for i in range(16)]
        self.__stiffnessOneMer = [np.zeros((6,6)) for i in range(4)]
        self.__stiffnessTwoMer = [np.zeros((18,18)) for i in range(16)] 
        self.__paramVec = np.zeros(1944)
        
        # Supplementary indices for format conversion
        self.__trilOneMer = np.tril_indices(6, -1)
        self.__trilTwoMer = np.tril_indices(18, -1)
        self.__triuOneMer = np.triu_indices(6)
        self.__triuTwoMer = np.triu_indices(18)
        
        # Tranformation matrix and model matrix/vector patterns
        sequenceLengths = IOHandler.read_sequence_information()
        maxSequenceLength = max(sequenceLengths)
        self.__setup_transformation_matrices()
        self.__setup_model_structure(maxSequenceLength)
        
        # Initialize data structures
        if isinstance(initArg, str):
            self.update_from_file(IOHandler, initArg)
        elif isinstance(initArg, np.ndarray):
            self.update_from_vector(initArg)
        else:
            raise TypeError("Unknown initialization argument type.")
    
    #-------------------------------------------------------------------------
    def __setup_model_structure(self, maxSequenceLength):
        """ Initializes index arrays for the cgDNA model matrix and vector
        
        This function contains one of the core routines of the cgDNA model.
        It pre-computes the indices of the monomer and dimer blocks in the 
        global model matrix and vector. Since all cgDNA models have the same
        pattern, the relevant locations can be computed for generic DNA 
        sequences up to a given length, which covers all sequences involved in
        the present computation.

        Parameters
        ----------
        maxSequenceLength : integer
            Maximum number of sequences to be expected for given sequence list
        """
        
        # Assembly buffers
        self.__vectorBuffer = np.zeros(12*maxSequenceLength-6)
        self.__matrixBufferOneMer = np.zeros(36*maxSequenceLength)
        self.__matrixBufferTwoMer = np.zeros(324*maxSequenceLength)
        
        # Index arrays
        self.__vecIndsOneMer = []
        self.__vecIndsTwoMer = []
        self.__matrixIndsOneMer = []
        self.__matrixIndsTwoMer = []
        
        # Compute indices of stand-alone 6x6 and 18x18 matrices
        rowIndsOneMer = []
        rowIndsTwoMer = []
        colIndsOneMer = []
        colIndsTwoMer = []
        rowArrayOneMer = np.ones(6, dtype=np.uint32)
        rowArrayTwoMer = np.ones(18, dtype=np.uint32)
        colArrayOneMer = np.array([i for i in range(6)], dtype=np.uint32)  
        colArrayTwoMer = np.array([i for i in range(18)], dtype=np.uint32)               
        for i in range(6):
            rowIndsOneMer.append(i*rowArrayOneMer)
            colIndsOneMer.append(colArrayOneMer)
            rowIndsTwoMer.append(i*rowArrayTwoMer)
            colIndsTwoMer.append(colArrayTwoMer)
        for i in range(6,18):
            rowIndsTwoMer.append(i*rowArrayTwoMer)
            colIndsTwoMer.append(colArrayTwoMer)           
        IndsOneMer = np.column_stack((np.hstack(rowIndsOneMer), 
                                      np.hstack(colIndsOneMer)))
        IndsTwoMer = np.column_stack((np.hstack(rowIndsTwoMer), 
                                      np.hstack(colIndsTwoMer)))
        
        # Compute index arrays from shift according to corresponding occurence
        # within sequence
        for i in range(maxSequenceLength):
            self.__vecIndsOneMer.append(np.arange(i*12, i*12+6))
            self.__vecIndsTwoMer.append(np.arange(i*12, i*12+18))
            self.__matrixIndsOneMer.append(IndsOneMer + i*12)
            self.__matrixIndsTwoMer.append(IndsTwoMer + i*12)
        
    #-------------------------------------------------------------------------
    def __setup_transformation_matrices(self):
        """ Assembles transformation matrix for the computation of 
            complementary monomers and dimers. The idea behind the
            transformation is that specific dimer/monomer groups have the 
            same physical properties, leading to (almost) equal parameters.
        """
        
        EMatrix = np.eye(6)
        EMatrix[0,0] = EMatrix[3,3] = -1
        self.__transformMatrixOneMer = EMatrix
        self.__transformMatrixTwoMer = np.zeros([18,18])
        self.__transformMatrixTwoMer[0:6,12:18] = EMatrix
        self.__transformMatrixTwoMer[6:12,6:12] = EMatrix
        self.__transformMatrixTwoMer[12:18,0:6] = EMatrix
    
    #-------------------------------------------------------------------------    
    def __complement_sequences(self):
        """ Computes complementary data structures
        
        Routine to reconstruct the full parameterset from two monomer and
        10 dimer arrays. The complementary arrays (4 monomer, 6 dimer) can be
        obtaines from simple algebraic manipulations involivng the transfor-
        mation matrix E. Here it is assumed that the encoding is such that the 
        dimers 5-10 are complementary to 11-16.
        """
        
        assert len(self.__sigmaOneMer) == 4,\
            "sigmaOneMer has wrong size."
        assert len(self.__sigmaTwoMer) == 16,\
            "sigmaTwoMer has wrong size."
        assert len(self.__stiffnessOneMer) == 4,\
            "stiffnessOneMer has wrong size."
        assert len(self.__stiffnessTwoMer) == 16,\
            "stiffnessTwoMer has wrong size."
        
        for i in range(2):
            self.__sigmaOneMer[i+2] = self.__transformMatrixOneMer \
                                      @ self.__sigmaOneMer[i]
            self.__stiffnessOneMer[i+2] = self.__transformMatrixOneMer \
                                          @ self.__stiffnessOneMer[i] \
                                          @ self.__transformMatrixOneMer  
        for i in range(4,10):
            self.__sigmaTwoMer[i+6] = self.__transformMatrixTwoMer \
                                       @ self.__sigmaTwoMer[i]
            self.__stiffnessTwoMer[i+6] = self.__transformMatrixTwoMer \
                                           @ self.__stiffnessTwoMer[i] \
                                           @ self.__transformMatrixTwoMer
                
    #-------------------------------------------------------------------------
    def check_postitive_definiteness(self):
        """ Checks if the given parameter set is "postive definite". This is 
            necessary for the set to be a valid candidate. The below criteria
            are sufficient.

        Returns
        ------
        Boolean for check result
        """

        assert len(self.__stiffnessOneMer) == 4,\
            "stiffnessOneMer has wrong size."
        assert len(self.__stiffnessTwoMer) == 16,\
            "stiffnessTwoMer has wrong size."
        
        # Criterion 1
        for i in range(10):
            bases = list(self.__twoMerKeys[i])
            alphaVal = self.__oneMerEncoding[bases[0]]
            betaVal = self.__oneMerEncoding[bases[1]]            
            matrix = self.__stiffnessTwoMer[i]
            
            testMatrix = np.copy(matrix.reshape((18,18)))
            testMatrix[0:6,0:6] +=\
                0.5 * self.__stiffnessOneMer[alphaVal].reshape((6,6))
            testMatrix[12:18,12:18] +=\
                0.5 * self.__stiffnessOneMer[betaVal].reshape((6,6))
            if not np.all(np.linalg.eigvals(testMatrix) > 0):
                return False
        
        # Criterion 2
        for i in range(16): 
            bases = list(self.__twoMerKeys[i])
            alphaVal = self.__oneMerEncoding[bases[0]]
            betaVal = self.__oneMerEncoding[bases[1]]            
            matrix = self.__stiffnessTwoMer[i]
            
            testMatrix = np.copy(matrix.reshape((18,18)))
            testMatrix[0:6,0:6] +=\
                self.__stiffnessOneMer[alphaVal].reshape((6,6))
            testMatrix[12:18,12:18] +=\
                0.5 * self.__stiffnessOneMer[betaVal].reshape((6,6))
            if not np.all(np.linalg.eigvals(testMatrix) > 0):
                return False
            
        return True
        
    #-------------------------------------------------------------------------
    def update_from_file(self, IOHandler, mode):   
        """ Reads cgDNA data structures from file and converts to vector
        
        Depending on the mode, the arrays are filled from cgDNA-optimized or
        proposal data (can also be the same). An independent group of monomers
        and dimers (one half of the symmetric matrices) is reduced to a 
        parameter vector.

        Parameters
        ----------
        IOHandler : IOHandler class obect
        mode : string
            Controls file to read from, can be "proposal" or "cgdna"
        """
        
        assert len(self.__sigmaOneMer) == 4,\
            "sigmaOneMer has wrong size."
        assert len(self.__sigmaTwoMer) == 16,\
            "sigmaTwoMer has wrong size."
        assert len(self.__stiffnessOneMer) == 4,\
            "stiffnessOneMer has wrong size."
        assert len(self.__stiffnessTwoMer) == 16,\
            "stiffnessTwoMer has wrong size."
        
        # Read data from file depending on mode
        self.__sigmaOneMer, self.__sigmaTwoMer,\
        self.__stiffnessOneMer, self.__stiffnessTwoMer\
            = IOHandler.read_cgDNA_data(mode) 
        for vector in self.__sigmaOneMer:
            assert len(vector) == 6, "Monomer vector has wrong size."
        for vector in self.__sigmaTwoMer:
            assert len(vector) == 18, "Dimer vector has wrong size."
        for matrix in self.__stiffnessOneMer:
            assert matrix.shape == (6,6), "Monomer matrix has wrong size."
        for matrix in self.__stiffnessTwoMer:
            assert matrix.shape == (18,18), "Dimer matrix has wrong size."
        
        # Collapse vectors and matrix triangles to parameter vector
        stiffnessOneMerReduced = [matrix[self.__triuOneMer] 
                           for matrix in self.__stiffnessOneMer[0:2]]
        stiffnessTwoMerReduced = [matrix[self.__triuTwoMer] 
                           for matrix in self.__stiffnessTwoMer[0:10]]     
        self.__paramVec = np.hstack((np.hstack(self.__sigmaOneMer[0:2]),
                                     np.hstack(self.__sigmaTwoMer[0:10]),
                                     np.hstack(stiffnessOneMerReduced),
                                     np.hstack(stiffnessTwoMerReduced)))       
        # Vectorize for later usage
        self.__stiffnessOneMer = [matrix.ravel() 
                                  for matrix in self.__stiffnessOneMer]
        self.__stiffnessTwoMer = [matrix.ravel() 
                                  for matrix in self.__stiffnessTwoMer]
       
    #-------------------------------------------------------------------------
    def update_from_vector(self, vector):
        """ Updates cgDNA data structures from a vector
        
        All matrices are reconstructed from that vector.

        Parameters
        ----------
        vector : 1D numpy float array
            Reduced cgDNA parameter vector
        """
        
        assert len(self.__sigmaOneMer) == 4,\
            "sigmaOneMer has wrong size."
        assert len(self.__sigmaTwoMer) == 16,\
            "sigmaTwoMer has wrong size."
        assert len(self.__stiffnessOneMer) == 4,\
            "stiffnessOneMer has wrong size."
        assert len(self.__stiffnessTwoMer) == 16,\
            "stiffnessTwoMer has wrong size."
        assert len(vector) == 1944, "Parameter vector has wrong size"
        
        # Assign to internal parameter vector
        self.__paramVec = vector

        # Reconstruct matrices
        for i in range(2):
            self.__sigmaOneMer[i] = vector[(i*6):((i+1)*6)]
            self.__stiffnessOneMer[i] =\
                self.__stiffnessOneMer[i].reshape((6,6))
            self.__stiffnessOneMer[i][self.__triuOneMer] \
                = vector[(192+i*21):(192+(i+1)*21)]
            self.__stiffnessOneMer[i][self.__trilOneMer] \
                = self.__stiffnessOneMer[i].T[self.__trilOneMer]        
        for i in range(10):
            self.__sigmaTwoMer[i] = vector[(12+i*18):(12+(i+1)*18)]
            self.__stiffnessTwoMer[i] =\
                self.__stiffnessTwoMer[i].reshape((18,18))
            self.__stiffnessTwoMer[i][self.__triuTwoMer] \
                = vector[(234+i*171):(234+(i+1)*171)]
            self.__stiffnessTwoMer[i][self.__trilTwoMer] \
                = self.__stiffnessTwoMer[i].T[self.__trilTwoMer]
        
        # Compute complementary structures 
        self.__complement_sequences()
        # Vectorize for later usage 
        self.__stiffnessOneMer = [matrix.ravel() 
                                  for matrix in self.__stiffnessOneMer]
        self.__stiffnessTwoMer = [matrix.ravel() 
                                  for matrix in self.__stiffnessTwoMer]
        
    #-------------------------------------------------------------------------
    def assemble_model(self, sequence):
        """ Assembly of the cgDNA model for a given sequence
        
        This function provides the core routine for the assembly of a cgDNA
        model. Given a sequence, it builds the global sigma vector and 
        stiffness matrix from the pre-computed index patterns. The assembly
        routine relies on the superposition of monomer and dimer blocks, whose
        index ranges have been pre-computed. The resulting model matrix is 
        sparse and returned in csc format.
    
        Parameters
        ----------
        sequence : List of Strings
            DNA sequence to assemble model for

        Returns
        -------
        modelMatrix : scipy sparse csc matrix object
            Sparse representation of the global cgDNA stiffness matrix
        sigmaVector : 1D numpy float array
            Global cgDNA sigma vector
        """
        
        seqLen = len(sequence)
        sigmaVector = self.__vectorBuffer[:12*seqLen-6]
        sigmaVector.fill(0)
        matrixInds =\
            np.vstack([np.vstack(self.__matrixIndsOneMer[:seqLen]),
                       np.vstack(self.__matrixIndsTwoMer[:seqLen-1])])
        matrixBufferOneMer = self.__matrixBufferOneMer[:36*seqLen]
        matrixBufferTwoMer = self.__matrixBufferTwoMer[:324*(seqLen-1)]       
        
        # Assemble stiffness matrix and model vector 
        for i in range(seqLen-1):
            oneMer = self.__oneMerEncoding[sequence[i]]
            twoMer = self.__twoMerEncoding[sequence[i] + sequence[i+1]]            
            matrixBufferOneMer[36*i:36*(i+1)] = self.__stiffnessOneMer[oneMer]
            matrixBufferTwoMer[324*i:324*(i+1)] = self.__stiffnessTwoMer[twoMer]
            sigmaVector[self.__vecIndsOneMer[i]] += self.__sigmaOneMer[oneMer]
            sigmaVector[self.__vecIndsTwoMer[i]] += self.__sigmaTwoMer[twoMer] 
            
        oneMer = self.__oneMerEncoding[sequence[-1]]
        matrixBufferOneMer[-36:] = self.__stiffnessOneMer[oneMer]
        sigmaVector[-6:] += self.__sigmaOneMer[oneMer]
        
        matrixData = np.hstack([matrixBufferOneMer, matrixBufferTwoMer])
        stiffnessMatrix =\
            csc_matrix((matrixData, (matrixInds[:,0],matrixInds[:,1])))
        stiffnessMatrix.sum_duplicates()         
        
        return stiffnessMatrix, sigmaVector
                       
    #-------------------------------------------------------------------------
    def get_parameter_vector(self):
        """ Get parameter vector
        
        Returns
        -------
        1D numpy float array
            Reduced parameter vector
        """
        
        return self.__paramVec
    
    #-------------------------------------------------------------------------
    def get_parameter_arrays(self):
        """ Get parameter arrays
        
        Returns
        -------
        Lists of 1/2D numpy float arrays
            Four lists of arrays for monomer and dimer parameters
        """
        
        return self.__sigmaOneMer, self.__sigmaTwoMer,\
               self.__stiffnessOneMer, self.__stiffnessTwoMer
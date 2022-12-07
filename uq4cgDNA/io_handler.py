##############################################################################
#                        _  _             ____  _   _    _                   #
#            _   _  __ _| || |   ___ __ _|  _ \| \ | |  / \                  #
#           | | | |/ _` | || |_ / __/ _` | | | |  \| | / _ \                 #
#           | |_| | (_| |__   _| (_| (_| | |_| | |\  |/ ___ \                #
#            \__,_|\__, |  |_|  \___\__, |____/|_| \_/_/   \_\               #
#                     |_|           |___/                                    #
#                                                                            #
##############################################################################

""" I/O Handler of the uq4cgDNA Framework
 
This module provides the functionalities for reading data from and writing
data to files. It consists of an IOHandler class whose objects are utilized for
the initialization of uq4cgDNA classes. Input reading capabilities include 
MD data, cgDNA parameter sets, prior and proposal distribution matrices and 
sequence information.
All input files are assumed to be formatted as txt files in the following form:
    - Sequences: One line, comma-separated list of strings
    - Shapes: Vertical concatenation of all column vectors
    - Stiffnesses: Vertical concatenation of all matrices
    
Additionally, the IOHandler provides output writing capabilities for logging
and result storage. The output files are again produced as txt files. Next to
a customized event handler, the IOhandler also provides the functionality of
controlling the logging.


Author Sebastian Krumscheidt, Maximilian Kruse
Date 06.06.2020
"""

#=========================== Preliminary Commands ============================
import os
import numpy as np
from scipy.sparse import csc_matrix
from contextlib import contextmanager


#============================= IOHandler Class ===============================
class IOHandler:
    """ I/O handler class for the uq4cgDNA framework.
    
    This class provides the functionalities for the io_handler module. Objects
    of this class can be given to the constructor of other uq4cgDNA classes to
    initialize their internal structures from files.
    
    Attributes
    ----------
    File names pointing to MD data : strings
    File names pointing to cgDNA optimization output: strings
    File names pointing to cgDNA proposal parameters: strings
    File names pointing to proposal and prior matrices: strings
    File names pointing to output and logging files: strings
    Logging flag: boolean
    
    Methods
    -------
    __init__(settings)
        Constructor from settings dictionary
    __assemble_from_file_static(fileName, numArrays, arraySize)
        Auxiliary routine for reading array lists of fixed structure
    __assemble_from_file_dynamic(fileName, sequenceList)
        Auxiliary routine for reading array lists of varying structure
    __read_array(fileName)
        Auxiliary routine for reading single array
    __read_sequence_list()
        Auxiliary routine for reading DNA sequence list
    read_encoding()
        Reads in monomer and dimer keys/order
    read_md_data(mode)
        Reads in data from MD simulations
    read_cgDNA_data(mode)
        Reads in parameter set from cgDNA optimization procedure or proposal
    read_prior_matrix()
        Reads in dense matrix for Gaussian prior
    read_proposal_matrix()
        Read in dense matrix for proposal covariance
    read_sequence_information()
        Reads in sequence lengths
    start_recording()
        Initialized context manager for output and log files
    log(message)
        Writes input to log file
    output(message)
        Writes input to output file
    """
    
    #-------------------------------------------------------------------------
    def __init__(self, settings):
        """ Constructor: Initializes file names for input and output

        Parameters
        ----------
        settings : dict
            Input and output file names/directories, logging/printing flags
        
        Raises
        ------
        KeyError
            Check if all mandatory settings are given
        """

        try:
            self.__fileMDSequences = settings["File_MD_Sequences"]
            self.__fileMDShapes = settings["File_MD_Shapes"]
            self.__fileMDStiffnesses = settings["File_MD_Stiffnesses"]
            
            self.__filecgDNAEncoding = settings["File_Encoding"]
            self.__filecgDNAOneMerSig = settings["File_cgDNA_OneMerSig"]
            self.__filecgDNATwoMerSig = settings["File_cgDNA_TwoMerSig"]
            self.__filecgDNAOneMerStiffness =\
                settings["File_cgDNA_OneMerStiffness"]        
            self.__filecgDNATwoMerStiffness =\
                settings["File_cgDNA_TwoMerStiffness"]

            self.__fileProposalOneMerSig =\
                settings["File_Proposal_OneMerSig"]
            self.__fileProposalOneMerStiffness =\
                settings["File_Proposal_OneMerStiffness"]
            self.__fileProposalTwoMerSig =\
                settings["File_Proposal_TwoMerSig"]
            self.__fileProposalTwoMerStiffness =\
                settings["File_Proposal_TwoMerStiffness"]
                
            self.__filePriorMatrix = settings["File_Prior_Matrix"]
            self.__fileProposalMatrix = settings["File_Proposal_Matrix"]
            
            self.__fileLogs = settings["File_Logs"]
            self.__fileOutput = settings["File_Output"]
            self.__enableLogging = settings["Logging"]
            self.__enablePrinting = settings["Printing"]

        except KeyError as ke:
            raise KeyError("Mandatory Setting " + str(ke) + 
                           " not correctly defined.")
            
    #-------------------------------------------------------------------------    
    def __assemble_from_file_static(self, fileName, numArrays, arraySize):
        """ Reads in lists of arrays of pre-defined size and list length.

        Parameters
        ----------
        fileName : string
            File to read from
        numArrays : integer
            Number of arrays to be expected in the file
        arraySize : integer
            Size of each array
            
        Raises
        ------
        FileNotFoundError
            Checks if file is existent
        IndexError
            Checks if correct sizes are given

        Returns
        -------
        arrayList : List of numpy float arrays
        """
        
        arrayList = []
        try:
            with open(fileName, 'r') as inFile:          
                for i in range(numArrays):
                    arrayList.append(np.loadtxt(inFile, delimiter=",", 
                                                max_rows=arraySize))
        except FileNotFoundError:
            raise FileNotFoundError("File " + fileName + " not found.")
        except IndexError:
            raise IndexError ("Index out of range while reading " + fileName)
                
        return arrayList
    
    #-------------------------------------------------------------------------
    def __assemble_from_file_dynamic(self, fileName, sequenceList):
        """ Reads in lists of arrays corresponding to DNA sequence list.
        
        The corresponding array size is assumed to be 12*n-6, where n is the
        length of the sequence.

        Parameters
        ----------
        fileName : string
            File to read from
        sequenceList : list of character lists
            Sequence list determining number and size of arrays
            
        Raises
        ------
        FileNotFoundError
            Checks if file is existent
        IndexError
            Checks if correct sizes are chosen

        Returns
        -------
        arrayList : list of numpy float arrays
        """
                
        arrayList = []
        try:
            with open(fileName, 'r') as inFile:          
                for i in range(len(sequenceList)):
                    arraySize = int(12 * len(sequenceList[i]) - 6)
                    arrayList.append(np.loadtxt(inFile, delimiter=",",
                                                max_rows=arraySize))
        except FileNotFoundError :
            raise FileNotFoundError("File " + fileName + " not found.")
        except IndexError:
            raise IndexError ("Index out of range while reading " + fileName)
                
        return arrayList
    
    #-------------------------------------------------------------------------
    def __read_array(self, fileName):
        """ Read in single array (matrix or vector) of arbitrary size
        
        Parameters
        ----------
        fileName : string
            File to read from
        
        Raises
        ------
        FileNotFoundError
            Checks if file is existent
        
        Returns
        -------
        Numpy array
        """
        
        try:
            with open(fileName, 'r') as inFile:
                array = np.loadtxt(inFile, delimiter=",")
        except FileNotFoundError:
            raise FileNotFoundError("File " + fileName + " not found.")
            
        return array
        
    #-------------------------------------------------------------------------
    def __read_sequence_list(self):
        """ Read in DNA sequence list
        
        The DNA sequence list is a list of string arrays that are not comma-
        separated. This procedure is performed here. The resulting structure
        is a list of lists containing the DNA bases for all sequences.
        
        Raises
        ------
        FileNotFoundError
            Checks if file is existent  
            
        Returns
        -------
        List of lists, containing strings
        """
        
        try:
            with open(self.__fileMDSequences, 'r') as inFile:
                stringArray = inFile.read().strip('\n')
        except FileNotFoundError:
            raise FileNotFoundError("File " + self.__fileMDSequences 
                                    + " not found.")    
            
        sequenceList = [list(sequence) for sequence in stringArray.split(',')]
        
        return sequenceList
    
    #-------------------------------------------------------------------------
    def read_encoding(self):
        """ Reads in the monomer/dimer encodings
        
        The encoding is the order of monomer/dimer letters corresponding to 
        the cgdna parameter set.
        
        Raises
        ------
        FileNotFoundError
            Checks if encoding file exists

        Returns
        -------
        oneMerKeys : List of strings
            Monomer encoding
        twoMerKeys : List of strings
            Dimer encoding
        """
        
        try:
            with open(self.__filecgDNAEncoding, 'r') as inFile:
                encLines = inFile.readlines()
        except FileNotFoundError:
            raise FileNotFoundError("File " + self.__filecgDNAEncoding 
                                    + " not found.")
            
        oneMerKeys = (encLines[0].strip()).split(',')
        twoMerKeys = (encLines[1].strip()).split(',')
        
        return oneMerKeys, twoMerKeys
           
    #-------------------------------------------------------------------------
    def read_md_data(self):
        """ Loads MD data from previously defined files.
                
        All vectors should have size 12*n-6 and all matrices
        (12*n-6)x(12*n-6), where n is the number of  sequences.

        Returns
        -------
        sequenceList : List of strings
            List of all sequences
        shapeVecs : List of 1D numpy float arrays
            MD shape vectors for all sequences
        stiffnessMatrices : List of 2D numpy float arrays
            Sparse MD stiffness matrices for all sequences in CSC format
        """
        
        sequenceList = self.__read_sequence_list() 
        shapeVecs = \
            self.__assemble_from_file_dynamic(self.__fileMDShapes, 
                                              sequenceList)       
        stiffnessMatrices = \
            self.__assemble_from_file_dynamic(self.__fileMDStiffnesses, 
                                              sequenceList)                                                                       
        stiffnessMatrices = [csc_matrix(matrix)
                             for matrix in stiffnessMatrices]
                
        return sequenceList, shapeVecs, stiffnessMatrices
        
    #-------------------------------------------------------------------------
    def read_cgDNA_data(self, mode):
        """ Reads the cgDNA parameter set from files.

        For the cgDNA Model the obtained vectors and matrices
        always have dimension 6/6x6 (monomer) and 18/18x18 (dimer).
        The files are assumed to contain all parameters, meaning that one 
        obtains four monomer and 16 dimer structures for each parameter. 
        The function has two different modes, namely 'cgdna' and 'proposal'.
        Depending on the mode, the data is read from the cgdna or proposal 
        files (these files might coincide).
        
        Parameters
        ----------
        mode : string
            'cgdna' or 'proposal'
            
        Raises
        ------
        ValueError
            Checks mode validity

        Returns
        -------
        sigmaOneMer : List of 1D numpy float arrays
            List of monomer sigma vectors
        sigmaTwoMer : List of 1D numpy float arrays
            List of dimer sigma vectors
        stiffnessOneMer : List of 2D numpy float arrays
            List of monomer K matrices
        stiffnessTwoMer : List of 2D numpy float arrays
            List of dimer K matrices
        """
        
        if mode == 'cgdna':
            sigmaOneMer = self.__assemble_from_file_static(
                          self.__filecgDNAOneMerSig, 4, 6)
            sigmaTwoMer = self.__assemble_from_file_static(
                          self.__filecgDNATwoMerSig, 16, 18)
            stiffnessOneMer = self.__assemble_from_file_static(
                              self.__filecgDNAOneMerStiffness, 4, 6)
            stiffnessTwoMer = self.__assemble_from_file_static(
                              self.__filecgDNATwoMerStiffness, 16, 18)
        elif mode == 'proposal':
            sigmaOneMer = self.__assemble_from_file_static(
                          self.__fileProposalOneMerSig, 4, 6)
            sigmaTwoMer = self.__assemble_from_file_static(
                          self.__fileProposalTwoMerSig, 16, 18)
            stiffnessOneMer = self.__assemble_from_file_static(
                              self.__fileProposalOneMerStiffness, 4, 6)
            stiffnessTwoMer = self.__assemble_from_file_static(
                              self.__fileProposalTwoMerStiffness, 16, 18)
        else:
            raise ValueError("Invalid mode for reading cgDNA data.")
        
        return sigmaOneMer, sigmaTwoMer, stiffnessOneMer, stiffnessTwoMer
    
    #-------------------------------------------------------------------------
    def read_prior_matrix(self):
        """ Reads in (dense) matrix for Gaussian prior.     

        Returns
        -------
        priorMatrix : 2D numpy float array
        """
        
        return self.__read_array(self.__filePriorMatrix)
    
    #-------------------------------------------------------------------------
    def read_proposal_matrix(self):
        """ Reads in (dense) matrix for proposal covariance     

        Returns
        -------
        covarianceMatrix : 2D numpy float array
        """
           
        return self.__read_array(self.__fileProposalMatrix)
            
    #-------------------------------------------------------------------------
    def read_sequence_information(self):
        """ Small auxiliary routine for prior information on MD sequences
        
        Returns
        -------
        seqLengths : List of integers
            Lengths of the given MD sequences
        """
        
        sequenceList = self.__read_sequence_list()  
        seqLengths = [len(sequence) for sequence in sequenceList]
        
        return seqLengths
    
    #-------------------------------------------------------------------------
    @contextmanager
    def start_recording(self):
        """ Context manager for output and logging
        
        This function provides a context manager for the output and logging 
        files for the uq4cgDNA computations. The opened files can be written
        to with the log() and output() functions given the context. The files
        are opened in append mode, so that existing data is not overwritten.
        Also note that the output of the computations is always written to a 
        files, whereas the activation of logging can be determined by the user.

        Raises
        ------
        OSError
            Checks for problems during file opening
        """
        
        try:
            if self.__enableLogging:              
                if os.path.isfile(self.__fileLogs):
                    print("Log file already exists, append new output.\n")
                self.__logFile = open(self.__fileLogs, 'a')
            if os.path.isfile(self.__fileOutput):
                print("Output file already exists, append new output.\n")
            self.__outputFile = open(self.__fileOutput, 'a')            
            yield
        except OSError as oe:
            raise OSError("Problem with opening file: " + str(oe))  
        finally:            
            self.__outputFile.close()
            if self.__enableLogging:
                self.__logFile.close()
    
    #-------------------------------------------------------------------------
    def log(self, message):
        """ uq4cgDNA logger
        
        String messages given to this file are written to the console and 
        (if logging is activated) to the specified logfile.

        Parameters
        ----------
        message : string
            Message to be logged/displayed

        Raises
        ------
        OSError
            Checks for error during I/O procedure
        """
        
        if self.__enablePrinting:
            print(message)
        
        if self.__enableLogging: 
            try:
                self.__logFile.write(message + "\n")
            except Exception:
                raise OSError("Could not write to logfile.")
            
    #-------------------------------------------------------------------------
    def output(self, resultArray):   
        """ uq4cgDNA output writer
        
        This function writes the given MCMC samples to a file. One result is
        stored per line. No whitespace is left between the lines.

        Parameters
        ----------
        resultArray : Given MCMC sample
            1D numpy float array

        Raises
        ------
        OSError
            Checks for error during writting procedure
        """
        
        try:
            np.savetxt(self.__outputFile, 
                       resultArray.reshape((1,1944)), 
                       delimiter=",")
        except Exception:
            raise OSError("Could not write to output file.")                
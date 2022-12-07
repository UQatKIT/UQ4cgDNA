##############################################################################
#                        _  _             ____  _   _    _                   #
#            _   _  __ _| || |   ___ __ _|  _ \| \ | |  / \                  #
#           | | | |/ _` | || |_ / __/ _` | | | |  \| | / _ \                 #
#           | |_| | (_| |__   _| (_| (_| | |_| | |\  |/ ___ \                #
#            \__,_|\__, |  |_|  \___\__, |____/|_| \_/_/   \_\               #
#                     |_|           |___/                                    #
#                                                                            #
##############################################################################

""" Test file of the uq4cgDNA Framework

This module contains the testing framework for the uq4cgDNA package. The test
automatization relies on the built-in "unittest" module, which facilitates a
test organisation in classes = TestCases. In here, the test cases have been 
written in accordance to the corresponding package modules, namely IOHandler,
cgDNAModel, Proposal, Posterior and MCMCSampler. Thus the setup does not 
distinguish between unit and regression tests, but rather relies on the 
structure of the underlying software package.
Please note that this is NOT a complete test suite aiming for full code 
coverage. It is rather a collection of intermediate results that can be used
for comprehension when modifying or extending the program. Extensions of the 
test code are of course highly welcome.
To run the test, simply execute this file with a python interpreter. Single
tests or test cases can be run by adding the corresponding names to the 'argv'
list at the bottom of this file.

IMPORTANT: Please be extremely careful when modifying the underlying data in 
           the 'TestData' directory. Changes in this data might corrupt
           multiple tests.

Author Sebastian Krumscheidt, Maximilian Kruse
Date 17.08.2020
"""

#=========================== Preliminary Commands ============================
import os
import sys
import unittest
import numpy as np
import scipy as sp
sys.path.append("../")

from uq4cgDNA import io_handler
from uq4cgDNA import cgdna_model
from uq4cgDNA import proposal
from uq4cgDNA import posterior
from uq4cgDNA import mcmc

#================================ Test Code ==================================

#-----------------------------------------------------------------------------
# For more information regarding the user input, please refer to the
# documentation/the main file.

# File Names
fileSettings = {
    # MD Data
    "File_MD_Sequences":                "TestData/Sequences.txt",
    "File_MD_Shapes":                   "TestData/Shapes_MD.txt",
    "File_MD_Stiffnesses":              "TestData/Stiffness_MD.txt",
    # cgDNA Data
    "File_Encoding":                    "TestData/Encoding.txt",
    "File_cgDNA_OneMerSig":             "TestData/OneMerSig.txt",
    "File_cgDNA_OneMerStiffness":       "TestData/OneMerStiff.txt",
    "File_cgDNA_TwoMerSig":             "TestData/TwoMerSig.txt",
    "File_cgDNA_TwoMerStiffness":       "TestData/TwoMerStiff.txt",
    "File_Proposal_OneMerSig":          "TestData/OneMerSig.txt",
    "File_Proposal_OneMerStiffness":    "TestData/OneMerStiff.txt",
    "File_Proposal_TwoMerSig":          "TestData/TwoMerSig.txt",
    "File_Proposal_TwoMerStiffness":    "TestData/TwoMerStiff.txt",
    # Hessian Matrix
    "File_Prior_Matrix":                "TestData/Hessian.txt",
    "File_Proposal_Matrix":             "TestData/Hessian.txt",
    # Test Data
    "File_Test_Sequence":               "TestData/TestSequence.txt",
    "File_Test_Shape":                  "TestData/TestShape.txt",
    "File_Test_Stiffness":              "TestData/TestStiffness.txt",
    # Output
    "File_Logs":                        "log.txt",
    "File_Output":                      "Output.txt",
    "Logging":                          True,
    "Printing":                         False
    }

# MCMC Sampler Settings
mcmcSettings = {
    "CN_Parameter":                     1e-3,
    "Number_of_Samples":                2,
    "Burn_in_Period":                   0,
    "Mean_Batch_Size":                  1,
    "Statistics_Interval":              1,
    "Output_Interval":                  1,
    }      


#============================= IOHandler Tests ===============================
class TestIOHandler(unittest.TestCase):
    """ IOHandler Test Case
    
    Tests
    -------
    test_encoding_read()
        Tests if Monomer/Dimer encoding is read in correctly
    test_md_read()
        Tests if MD sequance lists, shapes and stiffnesses are read correctly
    test_cgdna_read()
        Tests if cgDNA parameter vectors and matrices are read in correctly
    test_matrix_read()
        Tests if prior and proposal matrices are read in correctly
    test_info_read()
        Tests if MD sequence information is read in correctly
    """
    
    #-------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):
        """ Initialze IOHandler for tests """
        cls.__IOHandler = io_handler.IOHandler(fileSettings)
        
    #-------------------------------------------------------------------------
    def test_encoding_read(self):
        """ Tests reading of monomer/dimer encoding
        
        Read lists are compared to encoding entered by hand.
        """
        
        oneMerKeysTest = ['A','G','T','C']
        twoMerKeysTest = ['AT','GC','TA','CG','GT','TG','AG','GA',
                          'AA','GG','AC','CA','CT','TC','TT','CC']
        oneMerKeys, twoMerKeys = self.__IOHandler.read_encoding()
        
        self.assertEqual(oneMerKeys, oneMerKeysTest)
        self.assertEqual(twoMerKeys, twoMerKeysTest)

    #-------------------------------------------------------------------------
    def test_md_read(self):
        """ Tests reading of MD sequence lists, shapes and stiffnesses
        
        The data as read from file is compared to samples entered by hand.
        Three sequence lists are entered by hand. Additionally, shape and 
        stiffness arrays are elaborated from single entries and their norms
        computed in matlab.
        """
        
        MDSeqList, MDshapeVecs, MDstiffMats = self.__IOHandler.read_md_data()
        
        self.assertEqual(len(MDSeqList), 31)   
        for sequence in MDSeqList[:15]:
            self.assertEqual(len(sequence), 12)
        for sequence in MDSeqList[15:]:
            self.assertEqual(len(sequence), 24)
        self.assertEqual("".join(MDSeqList[0]),'AAGCAAACTAGC')          
        self.assertEqual("".join(MDSeqList[15]),'GCTTAGTTCAAATTTGAACTAAGC')
        self.assertEqual("".join(MDSeqList[30]),'GCCTAACCCTGCGCAGGGTTAGGC')
        
        self.assertEqual(len(MDshapeVecs), 31)
        for shapeVec in MDshapeVecs[:15]:
            self.assertEqual(shapeVec.shape, (138,))
        for shapeVec in MDshapeVecs[15:]:
            self.assertEqual(shapeVec.shape, (282,))
        self.assertAlmostEqual(np.linalg.norm(MDshapeVecs[0]), 
                               16.018239841599822, delta=1e-8)
        self.assertAlmostEqual(np.linalg.norm(MDshapeVecs[15]), 
                               22.935322012086854, delta=1e-8)
        self.assertAlmostEqual(np.linalg.norm(MDshapeVecs[30]), 
                               22.690646054476485, delta=1e-8)
        self.assertAlmostEqual(MDshapeVecs[0][0], 
                               1.181219693499127, delta=1e-8)
        self.assertAlmostEqual(MDshapeVecs[0][100], 
                               0.023011056958371, delta=1e-8)
        self.assertAlmostEqual(MDshapeVecs[15][0], 
                               0.691033711112585, delta=1e-8)
        self.assertAlmostEqual(MDshapeVecs[15][100], 
                               0.042914779589138, delta=1e-8)
        self.assertAlmostEqual(MDshapeVecs[30][0], 
                               0.859162103131543, delta=1e-8)
        self.assertAlmostEqual(MDshapeVecs[30][100], 
                               0.045205795104906, delta=1e-8)
        
        self.assertEqual(len(MDstiffMats), 31)
        for stiffMat in MDstiffMats[:15]:
            self.assertEqual(stiffMat.shape, (138,138))
        for stiffMat in MDstiffMats[15:]:
            self.assertEqual(stiffMat.shape, (282,282))
        self.assertAlmostEqual(sp.sparse.linalg.norm(MDstiffMats[0]), 
                               5.375188851941914e+02, delta=1e-8)
        self.assertAlmostEqual(sp.sparse.linalg.norm(MDstiffMats[15]), 
                               7.545276216670934e+02, delta=1e-8)
        self.assertAlmostEqual(sp.sparse.linalg.norm(MDstiffMats[30]), 
                               8.084405698794526e+02, delta=1e-8)
        self.assertAlmostEqual(MDstiffMats[0][0,0], 
                               6.486415280564866, delta=1e-8)
        self.assertAlmostEqual(MDstiffMats[0][99,101], 
                               2.079076036774629, delta=1e-8)
        self.assertAlmostEqual(MDstiffMats[15][0,0], 
                               9.338902042961708, delta=1e-8)
        self.assertAlmostEqual(MDstiffMats[15][99,101], 
                               1.051440216014669, delta=1e-8)
        self.assertAlmostEqual(MDstiffMats[30][0,0], 
                               9.059322340296072, delta=1e-8)
        self.assertAlmostEqual(MDstiffMats[30][99,101], 
                               1.467008330471772, delta=1e-8)
        
    #-------------------------------------------------------------------------    
    def test_cgdna_read(self):
        """ Tests reading of cgDNA parameter vectors and matrices
        
        The data as read from file is compared to samples entered by hand.
        The arrays are elaborated from single entries and their norms
        computed in Matlab.
        """
        
        sigOneMer, sigTwoMer, stiffOneMer, stiffTwoMer\
        = self.__IOHandler.read_cgDNA_data("cgdna")
        sigOneMerProp, sigTwoMerProp, stiffOneMerProp, stiffTwoMerProp\
        = self.__IOHandler.read_cgDNA_data("proposal")

        self.assertEqual(len(sigOneMer), 4)
        for sigVec in sigOneMer:
            self.assertEqual(sigVec.shape, (6,))
        self.assertTrue(np.allclose(sigOneMer[0], 
                        np.array([0.097429724733508, -4.949512132728251e-04, 
                                  0.999794839305910, 2.064948472000329, 
                                  1.470267207153262, 0.544163503314592])))   
        self.assertTrue(np.allclose(sigOneMer[1], 
                        np.array([0.042947285318994, 0.072710866116751,
                                  -0.535937960025677, -2.262429739173196,
                                  1.548845922583665, 0.248651613483678])))
        self.assertTrue(np.allclose(sigOneMer[2], 
                        np.array([-0.097429724733508, -4.949512132728251e-04,
                                  0.999794839305910, -2.064948472000329,
                                  1.470267207153262, 0.544163503314592])))
        self.assertTrue(np.allclose(sigOneMer[3], 
                        np.array([-0.042947285318994, 0.072710866116751,
                                  -0.535937960025677, 2.262429739173196,
                                  1.548845922583665, 0.248651613483678])))
        
        self.assertEqual(len(sigTwoMer), 16)
        for sigVec in sigTwoMer:
            self.assertEqual(sigVec.shape, (18,))
        self.assertAlmostEqual(np.linalg.norm(sigTwoMer[0]), 
                               3.388876566207226e+02, delta=1e-8)
        self.assertAlmostEqual(np.linalg.norm(sigTwoMer[10]), 
                               3.355250030254069e+02, delta=1e-8)    
        self.assertAlmostEqual(sigTwoMer[0][0], 
                               -98.412806401426320, delta=1e-8)
        self.assertAlmostEqual(sigTwoMer[0][10], 
                               -17.489902766856094, delta=1e-8)
        self.assertAlmostEqual(sigTwoMer[10][0], 
                               -97.456856475968240, delta=1e-8)
        self.assertAlmostEqual(sigTwoMer[10][10], 
                               -16.278176453240164, delta=1e-8)
        
        self.assertEqual(len(stiffOneMer), 4)
        for stiffMat in stiffOneMer:
            self.assertEqual(stiffMat.shape, (6,6))
        self.assertAlmostEqual(np.linalg.norm(stiffOneMer[0]), 
                               30.413294632510187, delta=1e-8)
        self.assertAlmostEqual(np.linalg.norm(stiffOneMer[3]), 
                               67.007770677021640, delta=1e-8)       
        self.assertAlmostEqual(stiffOneMer[0][0,0], 
                               0.311258871958889, delta=1e-8)
        self.assertAlmostEqual(stiffOneMer[0][4,3], 
                               -3.739134178666023, delta=1e-8)
        self.assertAlmostEqual(stiffOneMer[3][0,0], 
                               0.791591103444708, delta=1e-8)
        self.assertAlmostEqual(stiffOneMer[3][4,3], 
                               12.361390406656987, delta=1e-8)
        
        self.assertEqual(len(stiffTwoMer), 16)
        for stiffMat in stiffTwoMer:
            self.assertEqual(stiffMat.shape, (18,18))            
        self.assertAlmostEqual(np.linalg.norm(stiffTwoMer[0]), 
                               1.329333602642936e+02, delta=1e-8)
        self.assertAlmostEqual(np.linalg.norm(stiffTwoMer[10]), 
                               1.312826898304956e+02, delta=1e-8)          
        self.assertAlmostEqual(stiffTwoMer[0][0,0], 
                               9.963822994422173, delta=1e-8)
        self.assertAlmostEqual(stiffTwoMer[0][12,9], 
                               -0.834000468402614, delta=1e-8)
        self.assertAlmostEqual(stiffTwoMer[10][0,0], 
                               9.615028706661347, delta=1e-8)
        self.assertAlmostEqual(stiffTwoMer[10][12,9], 
                               -1.721955309117586, delta=1e-8)
        
        for i in range(4):
            self.assertTrue(np.allclose(sigOneMer[i], sigOneMerProp[i]))
            self.assertTrue(np.allclose(stiffOneMer[i], stiffOneMerProp[i]))
        for i in range(16):
            self.assertTrue(np.allclose(sigTwoMer[i], sigTwoMerProp[i]))
            self.assertTrue(np.allclose(stiffTwoMer[i], stiffTwoMerProp[i]))
        
    #-------------------------------------------------------------------------    
    def test_matrix_read(self):
        """ Tests reading of dense prior and proposal matrices
        
        The data as read from file is compared to samples entered by hand.
        The matrices are elaborated from single entries and their norms
        computed in matlab.
        """
        
        priorMatrix = self.__IOHandler.read_prior_matrix()
        proposalMatrix = self.__IOHandler.read_proposal_matrix()
        
        self.assertTrue(np.allclose(priorMatrix, proposalMatrix))
        self.assertEqual(priorMatrix.shape, (1944,1944))
        self.assertAlmostEqual(np.linalg.norm(priorMatrix), 
                               9.314508673868430e+03, delta=1e-8)
        self.assertAlmostEqual(priorMatrix[0,0], 
                               1.793950372796668e+02, delta=1e-8)
        self.assertAlmostEqual(priorMatrix[100,10], 
                               0.018228920415777, delta=1e-8)
        self.assertAlmostEqual(priorMatrix[100,1000], 
                               -0.010720849476549, delta=1e-8)
        
    #-------------------------------------------------------------------------    
    def test_info_read(self):
        """ Tests reading of MD sequence information
        
        The length of the test sequences is well-known, making the comparison
        trivial

        """
        seqLengths = self.__IOHandler.read_sequence_information()
        
        self.assertEqual(len(seqLengths), 31)   
        for seqLen in seqLengths[:15]:
            self.assertEqual(seqLen, 12)
        for seqLen in seqLengths[15:]:
            self.assertEqual(seqLen, 24)
   
        
#============================= cgDNAModel Tests ==============================
class TestcgDNAModel(unittest.TestCase):
    """ cgDNAModel Test Case
    
    Tests
    -------
    test_update()
        Tests if model is correctly updated from file and vector
    test_pd_check()
        Tests if positive definiteness check works on cgDNA parameterset
    test_assembly()
        Tests assembly of global model stiffness matrix and shape vector
    """
    
    #-------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):
        """ Initialze IOHandler for tests """
        cls.__IOHandler = io_handler.IOHandler(fileSettings)
    
    #-------------------------------------------------------------------------
    def setUp(self):
        """ Initialze cgDNAModels for tests """
        self.__cgDNAModel_1 = cgdna_model.cgDNAModel(self.__IOHandler, "cgdna")
        self.__cgDNAModel_2 = cgdna_model.cgDNAModel(self.__IOHandler, "cgdna")
        
    #-------------------------------------------------------------------------
    def test_update(self):
        """ Tests if model is correctly updated from file and vector
        
        The test is conducted by converting the structures back and forth 
        between full and reduced form and comparing the results.
        """
        
        paramVec = self.__cgDNAModel_1.get_parameter_vector()
        self.__cgDNAModel_2.update_from_vector(paramVec)
        
        sigOneMer_1, sigTwoMer_1, stiffOneMer_1, stiffTwoMer_1\
        = self.__IOHandler.read_cgDNA_data("cgdna")
        sigOneMer_2, sigTwoMer_2, stiffOneMer_2, stiffTwoMer_2\
        = self.__cgDNAModel_2.get_parameter_arrays()
        
        for i in range(4):
            self.assertTrue(np.allclose(sigOneMer_1[i], sigOneMer_2[i]))
            self.assertTrue(np.allclose(np.ravel(stiffOneMer_1[i]), 
                                                 stiffOneMer_2[i]))
        for i in range(16):
            self.assertTrue(np.allclose(sigTwoMer_1[i], sigTwoMer_2[i]))
            self.assertTrue(np.allclose(np.ravel(stiffTwoMer_1[i]), 
                                                 stiffTwoMer_2[i]))
    
    #-------------------------------------------------------------------------
    def test_pd_check(self):
        """ Tests if positive definiteness check works on cgDNA parameterset
        
        Note that this is not a general guarantee for the correctness of the 
        implementation. Further dcases with known result should be added in 
        the future.
        """
        
        self.assertTrue(self.__cgDNAModel_1.check_postitive_definiteness())
    
    #-------------------------------------------------------------------------
    def test_assembly(self):
        """ Tests assembly of global model stiffness matrix and shape vector
        
        The assembly is performed on a given test sequence and the result is 
        compared to the output of the original cgDNA code implemented in 
        Matlab.
        """
        
        with open("TestData/TestSequence.txt", 'r') as inFile:
                testSequence = inFile.read().strip('\n')
        testSequence = list(testSequence)
        testShape = np.loadtxt("TestData/TestShape.txt")
        testStiffness = np.loadtxt("TestData/TestStiffness.txt",delimiter=",")
        
        stiffnessMatrix, sigmaVector =\
            self.__cgDNAModel_1.assemble_model(testSequence)
        luDecomposition = sp.sparse.linalg.splu(stiffnessMatrix)
        stiffnessMatrix = stiffnessMatrix.todense()
        shapeVector = luDecomposition.solve(sigmaVector)
        
        self.assertTrue(np.allclose(testShape,shapeVector))
        self.assertTrue(np.allclose(testStiffness,stiffnessMatrix))

    
#============================== Posterior Tests ==============================
class TestPosterior(unittest.TestCase):
    """ Posterior Test Case
    
    Tests
    -------
    test_setup()
        Tests if relevant data structures are initialized correctly
    test_posterior_probability()
        Tests posterior probability computation for trivial use case
    """
    
    #-------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):
        """ Initialize IOHandler, cgDNAModel and Posterior for testing """
        cls.__IOHandler = io_handler.IOHandler(fileSettings)
        cls.__cgDNAModel = cgdna_model.cgDNAModel(cls.__IOHandler, "cgdna")
        cls.__Posterior = posterior.Posterior(cls.__IOHandler, 
                                              cls.__cgDNAModel)
        
    #-------------------------------------------------------------------------
    def test_setup(self):
        """ Tests if relevant data structures are initialized correctly
        
        The prior covariance and vector, as well as the MD stiffness matrix
        determinants are compared to data directly read from file.
        """
        
        priorMatrix, priorVector = self.__Posterior.get_prior_data()
        priorMatrixTest = self.__IOHandler.read_prior_matrix()
        priorMatrixTest = np.linalg.inv(priorMatrixTest)
        priorVectorTest = self.__cgDNAModel.get_parameter_vector()
        
        MDDeterminants = self.__Posterior.get_md_determinants()     
        _, _, MDMatricesTest = self.__IOHandler.read_md_data()
        MDMatricesTest = [matrix.todense() for matrix in MDMatricesTest]
        MDDeterminantsTest = [np.linalg.det(matrix) 
                              for matrix in MDMatricesTest]
        
        self.assertTrue(np.allclose(priorMatrix, priorMatrixTest))
        self.assertTrue(np.allclose(priorVector, priorVectorTest))
        
        for i in range(len(MDDeterminants)):
            self.assertAlmostEqual(1, MDDeterminantsTest[i]/MDDeterminants[i])
            
    #-------------------------------------------------------------------------
    def test_posterior_probability(self):
        """ Tests posterior probability computation for trivial use case
        
        The computation is tested for the trivial case of the initial cgDNA
        parameterset. Thus the test is not a guarantee for the correctness 
        of the algorithm and more elaborate test cases should be added in the
        future.
        """
        
        posteriorTest =\
            self.__Posterior.compute_posterior_probability(self.__cgDNAModel)
        self.assertAlmostEqual(posteriorTest, np.exp(-1), delta=1e-8)


#============================== Proposal Tests ===============================
class TestProposal(unittest.TestCase):
    """ Proposal Test Case
    
    Tests
    -------
    test_factories()
        Tests factory generation methods and correct data initialization
    test_candidate_generation()
        Tests generation of new sample candidates
    test_proposal_probability()
        Tests computation of conditional probability for candidates
    """
    
    #-------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):
        """
        Setup IOHandler, cgDNAModel, cn-parameter and proposal for testing
        Initialize (inverse) covariance matrix for comparison
        """
        
        cls.__IOHandler = io_handler.IOHandler(fileSettings)             
        cls.__cgDNAModel = cgdna_model.cgDNAModel(cls.__IOHandler, "cgdna")
        cls.__covMatrix = cls.__IOHandler.read_proposal_matrix()
        cls.__invCovMatrix = np.linalg.inv(cls.__covMatrix)
        cls.__cnParameter = mcmcSettings["CN_Parameter"]
        cls.__proposal = proposal.pCNProposal.from_file_cov(cls.__IOHandler, 
                                                            cls.__cnParameter)
        
    #-------------------------------------------------------------------------
    def test_factories(self):
        """ Tests factory generation methods and correct data initialization
        
        The proposal covariance (in form of a Cholesky decomposition) and its
        inverse are compared to data directly loaded from file.
        """
        
        proposal_1 = self.__proposal
        proposal_2 = proposal.pCNProposal.from_identity_cov(self.__cnParameter)
        
        covMatrixTest_1, invCovMatrixTest_1 \
            = proposal_1.get_covariance_matrices()
        covMatrixTest_1 = covMatrixTest_1 @ covMatrixTest_1.T
        covMatrixTest_2, invCovMatrixTest_2 \
            = proposal_2.get_covariance_matrices()
        covMatrixTest_2 = covMatrixTest_2.todense()
        invCovMatrixTest_2 = invCovMatrixTest_2.todense()
            
        self.assertTrue(np.allclose(covMatrixTest_1, 
                                       self.__covMatrix))
        self.assertTrue(np.array_equal(invCovMatrixTest_1, 
                                       self.__invCovMatrix))
        self.assertTrue(np.array_equal(covMatrixTest_2, np.identity(1944)))
        self.assertTrue(np.array_equal(invCovMatrixTest_2, np.identity(1944)))
        
    #-------------------------------------------------------------------------
    def test_candidate_generation(self):
        """ Tests generation of model candidate
        
        The new candidate is solely tested with respect to its shape. Thus
        this test is not a gurantee for the correctness of the algorithm and 
        more elaborate test cases should be deployed in the future.
        """
        
        oldCandidate = self.__cgDNAModel.get_parameter_vector()
        newCandidate = self.__proposal.generate(oldCandidate)
        
        self.assertEqual(oldCandidate.shape, newCandidate.shape)
        
    #-------------------------------------------------------------------------
    def test_proposal_probability(self):
        """ Tests computation of conditional proposal probability
        
        The probabilities are tested only regarding the feasible interval[0,1],
        since benchmark values are not available.
        """
        oldCandidate = self.__cgDNAModel.get_parameter_vector()
        newCandidate = self.__proposal.generate(oldCandidate)
        probIdentity =\
            self.__proposal.compute_conditional_probability(oldCandidate,
                                                            oldCandidate)
        probNew = self.__proposal.compute_conditional_probability(newCandidate,
                                                                  oldCandidate)
        self.assertLessEqual(probIdentity, 1)
        self.assertGreaterEqual(probIdentity, 1e-10)
        self.assertLessEqual(probNew, 1)
        self.assertGreaterEqual(probNew, 1e-10)


#============================ MCMCSampler Tests ==============================
class TestMCMCSampler(unittest.TestCase):
    """ MCMCSampler Test Case
    
    Tests
    -------
    test_setup()
        Tests correct initialization of data structures
    test_run()
        Tests the overall MCMC algorithm and resulting output files
    """
    
    #-------------------------------------------------------------------------
    @classmethod
    def setUpClass(cls):
        """ 
        Initialize IOHandler, cgDNAModel, Proposal, Posterior and Sampler
        for testing 
        """
        
        cls.__IOHandler = io_handler.IOHandler(fileSettings)
        cls.__cgDNAModel = cgdna_model.cgDNAModel(cls.__IOHandler, "cgdna")   
        cls.__Proposal =\
            proposal.pCNProposal.from_file_cov(cls.__IOHandler, 
                                               mcmcSettings["CN_Parameter"])    
        cls.__Posterior = posterior.Posterior(cls.__IOHandler, 
                                              cls.__cgDNAModel)
        cls.__cgDNAModel.update_from_file(cls.__IOHandler, "proposal")
        cls.__Sampler = mcmc.MCMCSampler(mcmcSettings, 
                                         cls.__cgDNAModel, 
                                         cls.__Posterior)
        
    def tearDown(self):
        """ Remove potential left-over files"""
        if os.path.isfile(fileSettings["File_Logs"]):
            os.remove(fileSettings["File_Logs"])
        if os.path.isfile(fileSettings["File_Output"]):
            os.remove(fileSettings["File_Output"])
        
    #-------------------------------------------------------------------------
    def test_setup(self):
        """ Tests correct initialization of data structures 
        
        The initial candidate vector and its posterior probability are compared
        to data directly loaded from file.
        """
        
        initCandidateTest = self.__cgDNAModel.get_parameter_vector()
        initProbTest =\
            self.__Posterior.compute_posterior_probability(self.__cgDNAModel)            
        initCandidate, initProb = self.__Sampler.get_status()
        
        self.assertEqual(initProb, initProbTest)
        self.assertTrue(np.array_equal(initCandidate, initCandidateTest))
        
    #-------------------------------------------------------------------------
    def test_run(self):
        """ Tests the overall MCMC algorithm and resulting output files
        
        This test can be seen as a regression test for the whole
        implementation. In the future the outcome of this test should be 
        compared to benchmark results. As of now, the resulting output file
        is check w.r.t. whether it contains the expected number of samples.
        """
        
        self.__Sampler.run(self.__IOHandler, self.__cgDNAModel,
                           self.__Proposal, self.__Posterior)  
        
        self.assertTrue(os.path.isfile(fileSettings["File_Logs"]))
        self.assertTrue(os.path.isfile(fileSettings["File_Output"]))
        
        with open(fileSettings["File_Output"], 'r') as inFile:
                resultArray = np.loadtxt(inFile, delimiter=",")
                
        self.assertEqual(resultArray.shape, 
                         (mcmcSettings["Number_of_Samples"],1944))
        

#-----------------------------------------------------------------------------
# Add classifier in argv list to run specific tests/ test cases
# Note: the 'ignored' parameter is added due to a particularity in the IPython
#       console and has to be removed for regular execution! 
if __name__ == '__main__':
    unittest.main(argv=['ignored', '-v'], exit=False)
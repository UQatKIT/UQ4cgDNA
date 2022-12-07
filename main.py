##############################################################################
#                        _  _             ____  _   _    _                   #
#            _   _  __ _| || |   ___ __ _|  _ \| \ | |  / \                  #
#           | | | |/ _` | || |_ / __/ _` | | | |  \| | / _ \                 #
#           | |_| | (_| |__   _| (_| (_| | |_| | |\  |/ ___ \                #
#            \__,_|\__, |  |_|  \___\__, |____/|_| \_/_/   \_\               #
#                     |_|           |___/                                    #
#                                                                            #
##############################################################################

""" uq4cgDNA main file

This is the main execution file for the uq4cgDNA software package. Technically 
this file is not part of the package itself, but can rather be seen as an 
example of its application. The file can be customized for specific purposes.
For more information regarding uq4cgDNA and the underlying implementation,
please refer to the readme file and documentation. These also contain a short
tutorial on the usage of the package.

Author Sebastian Krumscheidt, Maximilian Kruse
Date 21.08.2020
"""


#========================== Preliminary commands =============================
from uq4cgDNA import io_handler
from uq4cgDNA import cgdna_model
from uq4cgDNA import proposal
from uq4cgDNA import posterior
from uq4cgDNA import mcmc

#============================== Main function ================================
def main():
    
    # ------ User settings: I/O file names , Proposal and MCMC sampler -------
    
    # Files
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
    
    # Sampler
    mcmcSettings = {
        "Number_of_Samples":                3,
        "Burn_in_Period":                   0,
        "Mean_Batch_Size":                  5,
        "Statistics_Interval":              10,
        "Output_Interval":                  10,
        }
    
    # Proposal
    cnParameter = 1e-3

    # -------------------------- Program Execution ---------------------------
    try:
        print("#============================ uq4cgDNA "
              "=============================#\n")
        
        
        # ----------------------- Initialize model classes -------------------
        print(">>> Initialize ...\n")
        IOHandler = io_handler.IOHandler(fileSettings)      
        cgDNAModel = cgdna_model.cgDNAModel(IOHandler, "cgdna")   
        Proposal = proposal.pCNProposal.from_file_cov(IOHandler, cnParameter)    
        Posterior = posterior.Posterior(IOHandler, cgDNAModel)
        cgDNAModel.update_from_file(IOHandler, "proposal") 
        Sampler = mcmc.MCMCSampler(mcmcSettings, cgDNAModel, Posterior)
            
        # ------------------------------ Run Sampler -------------------------
        print(">>> Start MCMC run ...\n")
        Sampler.run(IOHandler, cgDNAModel, Proposal, Posterior)
    
    except KeyboardInterrupt:
        print("Computation aborted by user.")
    except AssertionError as AE:
        print("uq4cgDNA has raised an assertion:")
        print(AE)
    except Warning as W:
        print("uq4cgDNA has raised a warning:")
        print(W)
    except BaseException as E:
        print("uq4cgDNA has raised an exception:")
        print(E)
        
    finally:
        print("\n#================================"
              "===================================#")


if __name__ == "__main__":
    main()
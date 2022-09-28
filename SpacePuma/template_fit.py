# Author: Alex DelFranco
# Advisor: Rafa Martin Domenech
# Institution: Center for Astrophysics | Harvard & Smithsonian
# Date: 28 August 2022

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy as cp

from .fit_methods import fit_methods

class fit(fit_methods):
    def __init__(self,fit_module):

        # Tell the fit module that we are adding a gaussian fit
        fit_module.info['fit_dict']['--FIT NAME HERE--'] = self

        # Tell the fit module what parameters we want
        parameter_list = ['LIST PARAMS','HERE']

        # Set the model to be used
        self.model = self.NAMED_MODEL

        # Should there be a new axis added?
        self.new_ax = False

        # Setup the fit
        self.setup_fit(fit_module,parameter_list)

        return

    ##########################################
    ## THE MODEL
    ##########################################

    # MODEL TO BE REPLACED BY USER
    def NAMED_MODEL(self,xx,param1):
        # A function used for curve fitting
        y = xx
        return np.array(y)


    ##########################################
    ## OPTIONAL METHODS FOR EXTRA CONTROL
    ##########################################

    # ##########################################
    # ## PARAMETERS
    # ##########################################
    #
    # # Method to determine guess parameters and bounds
    # def get_params(self,parameters):
    #
    #     # Return the initial parameter guesses and bounds
    #     return(p0,bounds)
    #
    # ##########################################
    # ## PLOTTING METHODS
    # ##########################################
    #
    # # Method to add custom plotting for a certian fit
    # def plot_fit(self,fit_mod,stats,fit_number,ax,fit_ax=None):
    #
    #     return fit_line
    #
    # ##########################################
    # ## STATS
    # ##########################################
    #
    # # Method to add custom stats about a certain fit
    # def add_fit_stats(self,stats):
    #
    #     # Return a improved stats dict
    #     return(stats)

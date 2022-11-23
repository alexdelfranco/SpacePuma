# Author: Alex DelFranco
# Advisor: Rafa Martin Domenech
# Institution: Center for Astrophysics | Harvard & Smithsonian
# Date: 26 September 2022

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from scipy.optimize import curve_fit
import seaborn as sns

class fit_methods():

    def __init__(self):

        # List of parameter strings
        parameter_list = []

        # A dictionary is created to hold the parameters
        self.parameters = self.setup_parameters(parameter_list)

        # The model function below is saved as the model
        self.model = self.defined_model

        # New axis
        self.new_ax = False

        return

    ##########################################
    ## SETUP THE CLASS
    ##########################################

    def setup_fit(self,fit_module,parameter_list):

        # Create a fit parameters dictionary from the parameter list
        fit_module.info['fit_params'] = dict.fromkeys(parameter_list,{})
        for key in fit_module.info['fit_params']:
            fit_module.info['fit_params'][key] = {'Estimate':5,'Lower Bound':0,'Upper Bound':10}

        # Get the fit order from the info dictionary
        self.fit_order = fit_module.info['fit_order']

        # Calculate the parameters
        self.parameters = self.get_params(fit_module.info['fit_params'])

    ##########################################
    ## IMPORTED FITS
    ##########################################

    def fits(self): return

    ##########################################
    ## PARAMETERS
    ##########################################

    # Method to determine guess parameters and bounds
    def get_params(self,parameters):
        #Set initial parameters and bounds

        p0,ubound,lbound = [],[],[]

        for param in parameters:
            prms = parameters[param]
            p0.append(prms['Estimate'])
            lbound.append(prms['Lower Bound'])
            ubound.append(prms['Upper Bound'])

        return p0,[lbound,ubound]

    ##########################################
    ## THE MODEL
    ##########################################

    def defined_model(self,xx,*parameters): return xx

    ##########################################
    ## FITTING METHOD
    ##########################################

    def fit(self,fit_struct,ax,fit_info,new_ax=False):

        ##########################################
        ## PULL DATA
        ##########################################

        # Determine the primary curve for the current axis
        curve = self.artists_global['Primary Artists'][ax]
        # Separate the xdata values
        xdata = self.artists_global[ax][curve].get_data()[0]

        ##########################################
        ## DETERMINE WHICH FIT IS BEING PLOTTED
        ##########################################

        # Initialize a line number
        fit_number = 1
        # Determine which fit this is
        for key in self.artists[ax].keys():
            if 'Fit Line' in key: fit_number += 1

        # If there should be a new axis added
        if new_ax:
            ##########################################
            ## SETUP AXES
            ##########################################

            # Add an axis to the figure
            self.resize_figure(len(self.fig.axes)+1,self.fig)
            # Save that axis
            fit_ax = self.fig.axes[-1]
            self.data[ax]['Fit']['Axis'].append(fit_ax)

            ##########################################
            ## SETUP DICTIONARIES
            ##########################################

            # Add a plot on which to display the fit
            self.artists[fit_ax] = {}
            # Set the primary figure artist to the Fit Line
            self.artists_global['Primary Artists'][fit_ax] = 'Fit Line'
            # Create a data dictionary for the new axis
            self.data[fit_ax] = {
            'FitRange': self.data[ax]['FitRange'],
            'Data Axis':ax,
            'Fit Number': fit_number
            }
        else: fit_ax = None

        ##########################################
        ## FIT MODEL
        ##########################################

        # Fit n gaussians to the primary data on the selected plot
        stats = self.calculate_fit(ax,fit_struct.model,fit_struct.parameters)

        # Add optional extra statistics
        stats = fit_struct.add_fit_stats(stats)

        self.test = stats
        self.data[ax]['Fit'][f'Fit {fit_number}'] = {'Stats':stats}

        ##########################################
        ## PLOT THE MODEL
        ##########################################

        # Save the Fit Line data to the original axis
        self.data[ax]['Fit'][f'Fit {fit_number}']['Fit Line'] = {'xdata':xdata,'ydata':stats['simdat']}

        # Save the fit type and range to the original axis
        self.data[ax]['Fit'][f'Fit {fit_number}']['fit_type'] = fit_info['fit_type']
        self.data[ax]['Fit'][f'Fit {fit_number}']['fit_params'] = deepcopy(fit_info['fit_params'])
        self.data[ax]['Fit'][f'Fit {fit_number}']['fit_order'] = fit_info['fit_order']
        self.data[ax]['Fit'][f'Fit {fit_number}']['FitRange'] = self.data[ax]['FitRange']

        # Plot the optimal parameters
        self.artists[ax][f'Fit Line {fit_number}'] = fit_struct.plot_fit(self,stats,fit_number,ax,fit_ax)[0]


    ##########################################
    ## THE MODEL FIT
    ##########################################

    def calculate_fit(self,ax,model,parameters):

        ##########################################
        ## DEFINE DATA
        ##########################################

        # Determine the primary curve for the current axis
        curve = self.artists_global['Primary Artists'][ax]
        # Pull out the x and y data from that curve
        xdata,ydata = self.artists_global[ax][curve].get_data()
        # Determine the range of the fit
        xmin,xmax = sorted(self.data[ax]['FitRange'])

        p0,bounds = parameters

        ##########################################
        ## SUBSET THE DATA TO THE INPUT RANGE
        ##########################################

        # Determine the indices of the data values within the range
        ii = np.squeeze(np.where((xdata > xmin) & (xdata < xmax)))
        # Define subsets of the data arrays
        xsub,ysub = xdata[ii],ydata[ii]

        ##########################################
        ## FIT THE MODEL
        ##########################################

        # Fit a curve
        popt,pcov = curve_fit(model, xsub, ysub, p0=p0, bounds=bounds)

        ##########################################
        ## CALCULATE THE FIT STATS
        ##########################################

        # pcov is a matrix with values related to the errors of the fit.
        # To get the actual errors of the Gaussian parameters one needs to calculate the square root of the values in the diagonal

        # Create a fit from the xdata and fitted parameters
        simdat = model(xdata,*popt)
        # Calculate the errors of the fit
        perr = np.sqrt(np.diag(pcov))
        # Save the relevant statistics
        stats = {
        'popt':popt,'perr':perr,
        'xsub':xsub,'ysub':ysub,
        'xdata':xdata,
        'simdat':simdat
        }

        # Return the stats dict
        return(stats)

    ##########################################
    ## PLOT THE FIT
    ##########################################

    def plot_fit(self,fit_mod,stats,fit_number,ax,fit_ax=None):

        # Get the stats dictionary
        stats = fit_mod.data[ax]['Fit'][f'Fit {fit_number}']['Stats']
        xdata = stats['xdata']

        # Plot the fit line
        fit_line = ax.plot(xdata,stats['simdat'],lw=2,alpha=0.6,color='red')

        return fit_line

    ##########################################
    ## STATS
    ##########################################

    # Method to return stats about a certain fit
    def add_fit_stats(self,stats): return stats

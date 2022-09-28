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
        fit_module.info['fit_dict']['Gaussian'] = self

        # Tell the fit module what parameters we want
        parameter_list = ['Standard Dev','A','Mu']

        # Set the model to be used
        self.model = self.ngaussian_model

        # Should there be a new axis added?
        self.new_ax = True

        # Setup the fit
        self.setup_fit(fit_module,parameter_list)

        return

    ##########################################
    ## THE MODEL
    ##########################################

    def gaussian(self,xx,sigma,a,mu):
        # A gaussian function used for curve fitting
        exp = -(1/2) * ((xx-mu)/sigma) ** 2
        return np.array(a * np.exp(exp))

    # Define an n gaussian model
    def ngaussian_model(self,xx,*params):
        y = np.zeros_like(xx)
        for i in range(0,len(params),3):
            y = np.add(y, self.gaussian(xx,params[i],params[i+1],params[i+2]), casting="unsafe")
        # return(y + yoff + slope * (xx-xoff))
        return y

    ##########################################
    ## PARAMETERS
    ##########################################

    # Method to determine guess parameters and bounds
    def get_params(self,parameters):
        #Set initial parameters and bounds

        p0_temp,ubound_temp,lbound_temp = [],[],[]

        for param in parameters:
            prms = parameters[param]
            p0_temp.append(prms['Estimate'])
            lbound_temp.append(prms['Lower Bound'])
            ubound_temp.append(prms['Upper Bound'])

        p0,ubound,lbound = [],[],[]

        # Combine the separate parameter arrays
        for i in range(self.fit_order):
            p0 = np.concatenate((p0,p0_temp))
            lbound = np.concatenate((lbound,lbound_temp))
            ubound = np.concatenate((ubound,ubound_temp))
        bounds = np.concatenate(([lbound],[ubound]),axis=0)

        # Return the initial parameter guesses and bounds
        return(p0,bounds)

    ##########################################
    ## PLOTTING METHODS
    ##########################################

    def plot_fit(self,fit_mod,stats,fit_number,ax,fit_ax=None):

        ##########################################
        ## PLOT THE FIT LINE ON THE OG AXIS
        ##########################################

        # Get the stats dictionary
        stats = fit_mod.data[ax]['Fit'][f'Fit {fit_number}']['Stats']
        xdata = stats['xdata']
        popt = stats['popt']

        # Define an array of hex colors
        colors = sns.color_palette(fit_mod.style['color_palette']).as_hex()

        # Plot the fit line
        fit_line = ax.plot(xdata,stats['simdat'],lw=2,alpha=0.6,color='red')

        ##########################################
        ## PLOT THE FIT LINE ON THE NEW AXIS
        ##########################################

        if fit_ax is not None:
            # Create a total area variable
            total_area = 0
            # Loop through each fit gaussian and save and plot it
            for ii in np.arange(self.fit_order):
                self.test = 5692
                # Add the data for each gaussian
                ydata_g = self.gaussian(xdata, popt[3*ii], popt[3*ii + 1], popt[3*ii + 2])
                fit_mod.data[fit_ax][f'Fit {ii+1}'] = {'xdata':xdata,'ydata':ydata_g}
                # Calculate the area
                self.test = 56922352
                area = stats['area'][f'fit{ii+1}']
                # Add the area to the total area
                total_area += area
                self.test = 56922341
                err = stats['area_err'][f'fit{ii+1}_err']
                fit_mod.artists[fit_ax][f'Fit {ii+1}'] = fit_ax.fill_between(
                            fit_mod.data[fit_ax][f'Fit {ii+1}']['xdata'],
                            fit_mod.data[fit_ax][f'Fit {ii+1}']['ydata'],
                            label=f'Fit {ii+1} - Area: {round(area,3)}, Error: {round(err,3)}',
                            color=colors[ii%6],lw = 1,alpha=0.6)
                self.test = 5692234215

            self.test = 523523
            # Add a legend
            fit_ax.legend()
            # Set the legend on the original axis
            fit_line[0].set_label(f'Area: {round(total_area,3)}')
            ax.legend()

        return fit_line

    ##########################################
    ## STATS
    ##########################################

    # Method to return stats about a certain fit
    def add_fit_stats(self,stats):

        self.test = 903874
        area,area_err,fwhm,fwhm_err = {},{},{},{}
        popt,perr = stats['popt'],stats['perr']

        self.test = 903874234
        # Calculate the area of a Gaussian (for example, if you want to calculate the column density)
        for i in range(0,self.fit_order):
            # area of the first Gaussian: area = sqrt(2pi)*width*amp
            area[f'fit{i+1}'] = np.sqrt(2*np.pi)*popt[i*3]*popt[i*3 + 1]
            self.test = 903874123
            # error of the area of the first Gaussian calculated from the errors in the Gaussian parameters
            area_err[f'fit{i+1}_err'] = area[f'fit{i+1}'] * np.sqrt((perr[i*3]/popt[i*3])**2 + (perr[i*3 + 1]/popt[i*3 + 1])**2)

            self.test = stats
            # The FWHM (full width half maximum) of the Gaussian can be calculated from the width parameter of the Gaussian
            fwhm[f'fit{i+1}'] = 2.35482 * popt[3*i]
            fwhm_err[f'fit{i+1}_err'] = 2.35482 * perr[3*i]

        self.test = 90387423
        stats['area'],stats['area_err'],stats['fwhm'],stats['fwhm_err'] = area,area_err,fwhm,fwhm_err

        return(stats)

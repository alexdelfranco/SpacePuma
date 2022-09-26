# Author: Alex DelFranco
# Advisor: Rafa Martin Domenech
# Institution: Center for Astrophysics | Harvard & Smithsonian
# Date: 28 August 2022

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import functools

from scipy.optimize import curve_fit
import seaborn as sns

from base import widget_base

class fit(widget_base):
    def __init__(self,fig,menu=None,data=None,artists_global=None,data_global=None,load=None):
        '''
        '''
        self.widget_on = False
        # Setup the figure
        self.fig = fig

        # Initialize all global matplotlib artists
        self.artists_global = self.pull_artists(artists_global)
        # Initialize all global data
        self.data_global = self.pull_data(data_global)
        # Create a local artist dictionary
        if load is None: self.artists = {}
        else:
            # If load is not None, subscript it to the load fit dict
            load = load['fit']
            self.artists = load['artists']

        # Initialize defaults
        self.style,self.info,self.data = self.setup_defaults(load)

        # Initialize all buttons
        self.button_list,self.toggle_buttons = self.setup_buttons()
        # Place all the menu buttons
        if menu is None:
            self.menu = widgets.HBox()
            self.place_menu(self.menu,self.button_list)
        else: self.menu = menu

    def setup_defaults(self,load=None):

        if load is not None:
            load['info']['interactive_mode'] = 'off'
            return load['style'],load['info'],load['data']

        style = {
        'fill_color':'deepskyblue',
        'alpha':0.6,
        'vline_style':'--',
        'vline_width':2,
        'color_palette':'Set2'}

        info = {
        # Initialize the add, move, and delete
        'interactive_mode':'off',
        # Initialize an active axis
        'active_ax':None,
        # Initialize a selected boolean
        'selected':False,
        # Set an interactive click distance
        'click_dist':0.02,
        # Initialize a fit type
        'fit_type':'Gaussian',
        # Initialize a fit order
        'fit_order':1,
        # Initialize a fit std range
        'std_max':10
        }

        data = dict.fromkeys(self.artists_global['Interactive Axes'],self.data_init())

        return style,info,data

    def setup_buttons(self):

        ##########################################
        ## INTERACTIVE PLOTTING BUTTONS
        ##########################################

        # Add a button widget to add new points
        self.add_range_button = widgets.ToggleButton(description='Add Range')

        def on_add_range_button_clicked(b, self = self):

            # Turn off the other buttons
            for button in self.toggle_buttons:
                if button != self.add_range_button: button.value = False

            # Turn on add_range mode
            if self.info['interactive_mode'] == 'add_range':
                self.info['interactive_mode'] = 'off'
            else: self.info['interactive_mode'] = 'add_range'

        self.add_range_button.observe(functools.partial(on_add_range_button_clicked, self=self))

        #####################

        # Add a button widget to adjust current points
        self.adjust_range_button = widgets.ToggleButton(description='Adjust Range')

        def on_adjust_range_button_clicked(b, self = self):

            # Turn off the other buttons
            for button in self.toggle_buttons:
                if button != self.adjust_range_button: button.value = False

            # Turn on adjust_range mode
            if self.info['interactive_mode'] == 'adjust_range':
                self.info['interactive_mode'] = 'off'
            else: self.info['interactive_mode'] = 'adjust_range'

        self.adjust_range_button.observe(functools.partial(on_adjust_range_button_clicked, self=self))

        #####################

        # Add a selector widget to change the fit type
        self.fit_type_select = widgets.Select(description='Fit:',options=['Gaussian','Skew Gaussian'],rows=0,style={'description_width':'initial'},layout = widgets.Layout(width='200px'))

        def fit_type_select_clicked(b, self = self):

            self.info['Fit Type'] = self.fity_type_select.value

        self.fit_type_select.observe(functools.partial(fit_type_select_clicked, self=self))

        #####################

        # Add an integer text widget to change the fit order
        self.fit_order_inttext = widgets.IntText(description='Fit Order:',value=1,style={'description_width':'initial'},layout = widgets.Layout(width='100px'))

        def fit_order_inttext_clicked(b, self = self):

            self.info['fit_order'] = self.fit_order_inttext.value

        self.fit_order_inttext.observe(functools.partial(fit_order_inttext_clicked, self=self))

        #####################

        # Add a button widget to fit to the points
        self.fit_button = widgets.ToggleButton(description='Fit')

        def fit_button_clicked(b, self = self):

            # Turn off the other buttons
            for button in self.toggle_buttons:
                if button != self.fit_button: button.value = False

            # Turn on fit mode
            if self.info['interactive_mode'] == 'fit':
                self.info['interactive_mode'] = 'off'
            else: self.info['interactive_mode'] = 'fit'

        self.fit_button.observe(functools.partial(fit_button_clicked, self=self))

        #####################

        # Add a button widget to clear all points
        self.clear_axis_button = widgets.ToggleButton(description='Clear Plot')

        def clear_axis_button_clicked(b, self = self):

            # Turn off the other buttons
            for button in self.toggle_buttons:
                if button != self.clear_axis_button: button.value = False

            # Turn on clear mode
            if self.info['interactive_mode'] == 'clear':
                self.info['interactive_mode'] = 'off'
            else: self.info['interactive_mode'] = 'clear'

        self.clear_axis_button.observe(functools.partial(clear_axis_button_clicked, self=self))

        #####################

        return [self.add_range_button,self.adjust_range_button,self.fit_type_select,self.fit_order_inttext,self.fit_button,self.clear_axis_button],[self.add_range_button,self.adjust_range_button,self.fit_button,self.clear_axis_button]

    ##########################################
    ## UPDATE METHODS
    ##########################################

    # Method to create a new artist list
    def new_artists(self,ax,style):
        # Find the y axis minimum and maximum
        ymin,ymax = ax.get_ylim()

        artists = {
        'Bounds': ax.vlines([],ymin,ymax,linestyle=style['vline_style'],linewidth=style['vline_width']),
        'New Bound': ax.vlines([],ymin,ymax,linestyle=style['vline_style'],linewidth=style['vline_width'],color='red'),
        'Selected Bound': ax.vlines([],ymin,ymax,linestyle=style['vline_style'],linewidth=style['vline_width'],color='green')
        }
        return artists

    def data_init(self):
        return {
        'Range':[],
        'Fit':{'Line 1':{'xdata':[],'ydata':[]},'Axis':[]}
        }

    ##########################################
    ## EVENT HANDLER
    ##########################################

    def __call__(self,event):

        if not self.widget_on: return
        self.info['active_ax'] = None

        ##########################################
        ## UPDATE ARTISTS
        ##########################################

        for ax in self.fig.axes:
            # Determine which axis was clicked on
            if event.inaxes == ax: self.info['active_ax'] = ax
            # If the axis is not interactive, do not add new artists
            if ax not in self.artists_global['Interactive Axes']: continue
            # If the artist is interactive, initialize the appropriate artists
            self.update_dictionaries(ax,artist_key='Bounds',data_key='Fit')
            self.update(self.artists_global,self.data_global)

        # If a click is not within the axis, do nothing
        if self.info['active_ax'] == None: return
        else: ax = self.info['active_ax']

        ##########################################
        ## ADD RANGE
        ##########################################

        # If in add range mode
        if self.info['interactive_mode'] == 'add_range':

            # Check to see if the axis is an interactive axis
            if ax not in self.artists_global['Interactive Axes']: return
            # Check to see how many bounds are already in the axis
            if len(self.data[ax]['Range']) == 2: return

            # Clear all other points
            self.clear(self.artists[ax]['New Bound'],ax=ax,vlines=True)
            self.clear(self.artists[ax]['Selected Bound'],ax=ax,vlines=True)
            self.info['selected'] = False

            # Add the point to the array of boundaries
            self.data[ax]['Range'].append(event.xdata)

            # Replot all the points, including the extra one
            self.set_segments(self.artists[ax]['Bounds'],self.data[ax]['Range'],ax)

        ##########################################
        ## ADJUST RANGE
        ##########################################

        elif self.info['interactive_mode'] == 'adjust_range':

            # Check to see if the axis is an interactive axis
            if ax not in self.artists_global['Interactive Axes']: return

            # If a boundary is not selected
            if not self.info['selected']:

                # If there was a new boundary, clear it
                self.clear(self.artists[ax]['New Bound'],ax=ax,vlines=True)
                # Find the distance between the click and each boundary
                self.dat = event.xdata
                hdists = self.hdist(event.xdata,self.data[ax]['Range'],ax)
                self.hdists = hdists
                # Find the index of the point closest to the click
                self.info['close_bound'] = np.nanargmin(hdists)

                # If the point is close to the click
                if hdists[self.info['close_bound']] < self.info['click_dist']:

                    # Replot the selected boundary in a different color
                    self.set_segments(self.artists[ax]['Selected Bound'],[self.data[ax]['Range'][self.info['close_bound']]],ax)
                    # State that a point has been selected
                    self.info['selected'] = True

            # If a point has already been selected
            else:
                # Remove the bound from the data array
                self.data[ax]['Range'] = np.delete(self.data[ax]['Range'],self.info['close_bound'])
                # Remove the temporary plotted point
                self.clear(self.artists[ax]['Selected Bound'],ax=ax,vlines=True)

                # Plot the new line
                self.set_segments(self.artists[ax]['New Bound'],[event.xdata],ax)
                # Add the point to the array of points
                self.data[ax]['Range'] = np.append(self.data[ax]['Range'],event.xdata)

                # Replot all points
                self.set_segments(self.artists[ax]['Bounds'],self.data[ax]['Range'],ax)

                # No point is currently selected
                self.info['selected'] = False

        ##########################################
        ## MODEL FIT
        ##########################################

        # If in fit mode
        elif self.info['interactive_mode'] == 'fit':

            # Check to see if the axis is an interactive axis
            if ax not in self.artists_global['Interactive Axes']: return

            # Only fit if there are two boundaries
            if len(self.data[ax]['Range']) < 2: return

            # Add an axis to the figure
            self.resize_figure(len(self.fig.axes)+1,self.fig)
            # Save that axis
            fit_ax = self.fig.axes[-1]
            self.data[ax]['Fit']['Axis'].append(fit_ax)

            # Add a plot on which to display the fit
            self.artists[fit_ax] = {}
            # Define an array of hex colors
            colors = sns.color_palette(self.style['color_palette']).as_hex()

            # Create a data dictionary for the new axis
            self.data[fit_ax] = {
            'Range': self.data[ax]['Range'],
            'Data Axis':ax,
            'Stats': {},
            'Fit Line':{'xdata':[],'ydata':[]},
            'Fit Number': 1
            }

            # Determine the primary curve for the current axis
            curve = self.artists_global['Primary Artists'][ax]
            # Separate the xdata values
            xdata = self.artists_global[ax][curve].get_data()[0]

            #######
            model = self.ngaussian_model; gnumber = self.info['fit_order']

            # Fit n gaussians to the primary data on the selected plot
            self.data[fit_ax]['Stats'] = self.calc_fit(ax,model,gnumber)

            popt = self.data[fit_ax]['Stats']['popt']
            if self.info['fit_type'] == 'Gaussian': self.plot_gaussian_fit(xdata,popt,colors,ax,fit_ax)

            fit_ax.legend()
            ax.legend()

            # Set the primary figure artist to the Fit Line
            self.artists_global['Primary Artists'][fit_ax] = 'Fit Line'

        ##########################################
        ## CLEAR AXIS
        ##########################################

        # If in clear mode
        elif self.info['interactive_mode'] == 'clear':

            # If the artist is not meant for data display
            if ax in self.artists_global['Interactive Axes']:
                # If there is a fit axis
                if len(self.data[ax]['Fit']['Axis']) != 0:
                    # For each fit axis
                    for fit_ax in self.data[ax]['Fit']['Axis']:
                        # Clear data
                        del self.data[fit_ax]
                        # Clear global data
                        if fit_ax in self.data_global.keys():
                            del self.data_global[fit_ax]
                        # Clear artists
                        for artist in self.artists[fit_ax]:
                            self.artists[fit_ax][artist].set_paths([])
                        # Clear the artist dictionary entry
                        del self.artists[fit_ax]
                        # Clear global artists
                        if fit_ax in self.artists_global.keys():
                            del self.artists_global[fit_ax]
                        # Clear the primary artists
                        del self.artists_global['Primary Artists'][fit_ax]

                        # Delete axis
                        self.fig.delaxes(fit_ax)
                        self.resize_figure(len(self.fig.axes),self.fig)

                # Clear the data dictionary
                self.data[ax] = self.data_init()
                # Clear artists
                self.set_segments(self.artists[ax]['Bounds'],[],ax)
                self.set_segments(self.artists[ax]['New Bound'],[],ax)
                self.set_segments(self.artists[ax]['Selected Bound'],[],ax)

                for artist in list(self.artists[ax].keys()):
                    if 'Fit Line' in artist:
                        self.artists[ax][artist].set_data([],[])
                        del self.artists[ax][artist]

            else:
                # Clear global artists
                if ax in self.artists_global.keys():
                    del self.artists_global[ax]
                # Clear artists on parent plot
                for artist in list(self.artists[self.data[ax]['Data Axis']].keys()):
                    if artist == f"Fit Line {self.data[ax]['Fit Number']}":
                        self.test = artist
                        # Clear line data
                        self.artists[self.data[ax]['Data Axis']][artist].set_data([],[])
                        del self.artists[self.data[ax]['Data Axis']][artist]
                        if self.data[ax]['Data Axis'] in self.artists_global.keys():
                            # if artist in self.artists_global[self.data[ax]['Data Axis']].keys():
                            self.artists_global[self.data[ax]['Data Axis']].pop(artist,None)

                # Clear artists
                del self.artists[ax]
                # Clear the data on the parent plot
                self.data[self.data[ax]['Data Axis']]['Fit'] = {'Line 1':{'xdata':[],'ydata':[]},'Axis':[]}
                # Clear the data on the parent plot
                if self.data[ax]['Data Axis'] in self.data_global.keys():
                    self.data[self.data[ax]['Data Axis']]['Fit'] = {'Line 1':{'xdata':[],'ydata':[]},'Axis':[]}
                # Clear data
                del self.data[ax]
                # Clear global data
                if ax in self.data_global.keys():
                    del self.data_global[ax]
                # Clear the primary artists
                del self.artists_global['Primary Artists'][ax]
                # Delete axis
                self.fig.delaxes(ax)
                self.resize_figure(len(self.fig.axes),self.fig)

            # Clear the legend
            ax.get_legend().remove()

        ##########################################
        ## UPDATE ARTISTS
        ##########################################

        # Update artists
        plt.show()

    ##########################################
    ## FITTING METHODS
    ##########################################

    def calc_fit(self,ax,model,gnumber):

        # Determine the primary curve for the current axis
        curve = self.artists_global['Primary Artists'][ax]
        # Pull out the x and y data from that curve
        xdata,ydata = self.artists_global[ax][curve].get_data()
        # Determine the range of the fit
        xmin,xmax = sorted(self.data[ax]['Range'])
        # Determine the max y value of the fit
        ymax = np.max(ydata)

        # # Retrieve the fit order
        # gnumber = self.info['fit_order']
        # Determine the input parameters and bounds
        p0,bounds = self.get_params(xmin,xmax,ymax,gnumber)

        # Determine the indices of the data values within the range
        ii = np.squeeze(np.where((xdata > xmin) & (xdata < xmax)))
        # Define subsets of the data arrays
        xsub,ysub = xdata[ii],ydata[ii]

        # Fit a curve
        popt,pcov = curve_fit(model, xsub, ysub, p0=p0, bounds=bounds)

        ## Fit products

        # Create a fit from the xdata and fitted parameters
        simdat = model(xdata,*popt)

        # pcov is a matrix with values related to the errors of the fit.
        # To get the actual errors of the Gaussian parameters one needs to calculate the square root of the values in the diagonal

        # Calculate the errors of the fit
        perr = np.sqrt(np.diag(pcov))

        # Save the relevant statistics
        stats = {
        'popt':popt,'perr':perr,
        'xsub':xsub,'ysub':ysub,
        'simdat':simdat
        }

        stats = self.add_fit_stats(stats,gnumber)

        return(stats)

    def plot_gaussian_fit(self,xdata,popt,colors,ax,fit_ax):
        # Initialize a line number
        fnum = 1
        # Determine which fit this is
        for key in self.artists[ax].keys():
            if 'Fit Line' in key: fnum += 1
        if fnum != 1: self.data[fit_ax]['Fit Number'] = fnum

        # Save the Fit Line data to the original axis
        self.data[ax]['Fit'][f'Line {fnum}'] = {'xdata':[],'ydata':[]}
        self.data[ax]['Fit'][f'Line {fnum}']['xdata'] = xdata
        self.data[ax]['Fit'][f'Line {fnum}']['ydata'] = self.data[fit_ax]['Stats']['simdat']
        # Save the Fit Line data to the original axis
        self.data[fit_ax]['Fit Line']['xdata'] = xdata
        self.data[fit_ax]['Fit Line']['ydata'] = self.data[fit_ax]['Stats']['simdat']
        # Create a total area variable
        total_area = 0
        # Loop through each fit gaussian and save and plot it
        for ii in np.arange(self.info['fit_order']):
            self.data[fit_ax][f'Fit {ii+1}'] = {'xdata':[],'ydata':[]}
            self.data[fit_ax][f'Fit {ii+1}']['xdata'] = xdata
            self.data[fit_ax][f'Fit {ii+1}']['ydata'] = self.gaussian(xdata, popt[3*ii], popt[3*ii + 1], popt[3*ii + 2])
            # Calculate the area
            area = self.data[fit_ax]['Stats']['area'][f'fit{ii+1}']
            # Add the area to the total area
            total_area += area
            err = self.data[fit_ax]['Stats']['area_err'][f'fit{ii+1}_err']
            self.artists[fit_ax][f'Fit {ii+1}'] = fit_ax.fill_between(self.data[fit_ax][f'Fit {ii+1}']['xdata'],self.data[fit_ax][f'Fit {ii+1}']['ydata'],label=f'Fit {ii+1} - Area: {round(area,3)}, Error: {round(err,3)}',color = colors[ii%6], lw = 1,alpha=0.6)

            # Plot the fit line
            self.artists[ax][f'Fit Line {fnum}'] = ax.plot(self.data[ax]['Fit'][f'Line {fnum}']['xdata'],self.data[ax]['Fit'][f'Line {fnum}']['ydata'],label=f'Area: {round(total_area,3)}',lw=2,alpha=0.6,color='red')[0]

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
    def get_params(self,xmin,xmax,ymax,gnum=1):
        #Set initial parameters and bounds

        xavg = (xmin+xmax)/2
        std_max = self.info['std_max']

        p0,bound1,bound2 = [],[],[]

        #             std        a        mu
        p0_fit   =  [ std_max/2, ymax,  xavg ]   # GUESSES
        bnd_fit1 =  [ 0,         1e-6,    xmin ]  # LOWER BOUNDS
        bnd_fit2 =  [ std_max,   ymax*2, xmax ]  # UPPER BOUNDS

        # Combine the separate parameter arrays
        for i in range(gnum):
            p0 = np.concatenate((p0,p0_fit))
            bound1 = np.concatenate((bound1,bnd_fit1))
            bound2 = np.concatenate((bound2,bnd_fit2))
        bounds = np.concatenate(([bound1],[bound2]),axis=0)

        # Return the initial parameter guesses and bounds
        return(p0,bounds)

    ##########################################
    ## STATS
    ##########################################

    # Method to return stats about a certain fit
    def add_fit_stats(self,stats,gnum):

        area,area_err,fwhm,fwhm_err = {},{},{},{}
        popt,perr = stats['popt'],stats['perr']

        # Calculate the area of a Gaussian (for example, if you want to calculate the column density)
        for i in range(0,gnum):
            # area of the first Gaussian: area = sqrt(2pi)*width*amp
            area[f'fit{i+1}'] = np.sqrt(2*np.pi)*popt[i*3]*popt[i*3 + 1]
            # error of the area of the first Gaussian calculated from the errors in the Gaussian parameters
            area_err[f'fit{i+1}_err'] = area[f'fit{i+1}'] * np.sqrt((perr[i*3]/popt[i*3])**2 + (perr[i*3 + 1]/popt[i*3 + 1])**2)

            # The FWHM (full width half maximum) of the Gaussian can be calculated from the width parameter of the Gaussian
            fwhm[f'fit{i+1}'] = 2.35482 * popt[3*i]
            fwhm_err[f'fit{i+1}_err'] = 2.35482 * perr[3*i]

        stats['area'],stats['area_err'],stats['fwhm'],stats['fwhm_err'] = area,area_err,fwhm,fwhm_err

        return(stats)

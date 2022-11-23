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
import copy as cp

from .base import widget_base
from .fit_methods import fit_methods

from .gaussian_fit import fit as gaussian_fit

class fit(widget_base,fit_methods):

    def fits(self):
        gaussian_fit(self)

    def __init__(self,fig,menu=None,param_menu=None,data=None,artists_global=None,data_global=None,load_data=None):
        '''
        '''
        # Initialize the widget in the off position
        self.widget_on = False
        # Set the display setting to display output
        self.show = True
        # Setup the figure
        self.fig = fig
        # Setup possible load data
        if load_data is not None:
            if 'fit' in load_data: load_data = load_data['fit']

        # Initialize all global matplotlib artists
        self.artists_global = self.pull_artists(artists_global)
        # Initialize all global data
        self.data_global = self.pull_data(data_global)
        # Create a local artist dictionary
        self.artists = {}

        # Initialize defaults
        self.style,self.info,self.data = self.setup_defaults(load_data)

        # Setup the fits
        self.fits()

        # Initialize all buttons
        self.button_list,self.toggle_buttons,self.param_button_list = self.setup_buttons()
        # Place all the menu buttons
        if menu is None:
            self.menu = widgets.HBox()
            self.place_menu(self.menu,self.button_list)
        else: self.menu = menu
        if param_menu is None:
            self.param_menu = widgets.HBox()
            self.place_menu(self.param_menu,self.param_button_list)
        else: self.param_menu = param_menu

        if load_data is not None: self.load(load_data)

    def setup_defaults(self,load_data=None):

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
        # Initialize a list of fit types
        'fit_dict':{},
        # Initialize a parameter:
        'parameter':'',
        # Initialize a fit type
        'fit_type':'None',
        # Initialize a parameters dict
        'fit_params':{},

        ##### TO BE REMOVED
        # Initialize a fit order
        'fit_order':1,
        # Initialize a fit std range
        'std_max':10
        }

        data = dict.fromkeys(self.artists_global['Interactive Axes'],self.data_init())

        if load_data is not None:
            load_data['info']['interactive_mode'] = 'off'
            load_data['info']['selected'] = False
            load_data['info']['active_ax'] = None

            return load_data['style'],load_data['info'],data

        return style,info,data

    def setup_parameters(self,params):
        param_dict = {}
        for param in params:
            param_dict[param] = {
            'Estimate':0,
            'Lower Bound':0,
            'Upper Bound':0
            }
        return param_dict

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
        self.fit_type_select = widgets.Select(description='Fit:',options=list(np.concatenate([['None'],list(self.info['fit_dict'].keys())])),rows=0,style={'description_width':'initial'},layout = widgets.Layout(width='200px'))

        def fit_type_select_clicked(b, self = self):

            self.info['fit_type'] = self.fit_type_select.value

            if self.fit_type_select.value == 'None':
                # Clear the parameter menu
                self.place_menu(self.param_menu,[])
                return

            if len(self.info['fit_params'].keys()) > 0:
                # Display the paramter menu
                self.place_menu(self.param_menu,self.param_button_list)
                self.info['parameter'] = self.parameter_select.value

        self.fit_type_select.observe(functools.partial(fit_type_select_clicked, self=self))

        #####################

        # Add an integer text widget to change the fit order
        self.fit_order_inttext = widgets.IntText(description='Order:',value=1,style={'description_width':'initial'},layout = widgets.Layout(width='95px'))

        def fit_order_inttext_clicked(b, self = self):

            # Save the fit order
            self.info['fit_order'] = self.fit_order_inttext.value
            # Retrieve the fit module
            fit_module = self.info['fit_dict'][self.info['fit_type']]
            # Set the fit order in the module
            fit_module.fit_order = self.info['fit_order']
            # Recalculate the parameters
            fit_module.parameters = fit_module.get_params(self.info['fit_params'])

        self.fit_order_inttext.observe(functools.partial(fit_order_inttext_clicked, self=self))

        #####################

        # Add a selector widget to change the input paramters
        self.parameter_select = widgets.Select(description='Paramters:',options=list(self.info['fit_params'].keys()),rows=0,style={'description_width':'initial'},layout = widgets.Layout(width='200px'))

        def parameter_select_clicked(b, self = self):

            self.info['parameter'] = self.parameter_select.value

            # Retrieve the current values of the fit parameters
            self.estimate_inttext.value = self.info['fit_params'][self.info['parameter']]['Estimate']
            self.lower_bound_inttext.value = self.info['fit_params'][self.info['parameter']]['Lower Bound']
            self.upper_bound_inttext.value = self.info['fit_params'][self.info['parameter']]['Upper Bound']

        self.parameter_select.observe(functools.partial(parameter_select_clicked, self=self))

        #####################

        # Add an integer text widget to set a paramter estimate
        self.estimate_inttext = widgets.IntText(description='Estimate:',value=5,style={'description_width':'initial'},layout = widgets.Layout(width='120px'))

        def estimate_inttext_clicked(b, self = self):

            # Save the fit parameter
            self.info['fit_params'][self.info['parameter']]['Estimate'] = cp.deepcopy(self.estimate_inttext.value)
            # Retrieve the fit module
            fit_module = self.info['fit_dict'][self.info['fit_type']]
            # Recalculate the parameters
            fit_module.parameters = fit_module.get_params(self.info['fit_params'])

        self.estimate_inttext.observe(functools.partial(estimate_inttext_clicked, self=self))

        #####################

        # Add an integer text widget to set the lower paramter bound
        self.lower_bound_inttext = widgets.IntText(description='Lower Bound:',value=0,style={'description_width':'initial'},layout = widgets.Layout(width='150px'))

        def lower_bound_inttext_clicked(b, self = self):

            # Save the fit parameter
            self.info['fit_params'][self.info['parameter']]['Lower Bound'] = cp.deepcopy(self.lower_bound_inttext.value)
            # Retrieve the fit module
            fit_module = self.info['fit_dict'][self.info['fit_type']]
            # Recalculate the parameters
            fit_module.parameters = fit_module.get_params(self.info['fit_params'])

        self.lower_bound_inttext.observe(functools.partial(lower_bound_inttext_clicked, self=self))

        #####################

        # Add an integer text widget to set the upper paramter bound
        self.upper_bound_inttext = widgets.IntText(description='Upper Bound:',value=10,style={'description_width':'initial'},layout = widgets.Layout(width='150px'))

        def upper_bound_inttext_clicked(b, self = self):

            # Save the fit parameter
            self.info['fit_params'][self.info['parameter']]['Upper Bound'] = cp.deepcopy(self.upper_bound_inttext.value)
            # Retrieve the fit module
            fit_module = self.info['fit_dict'][self.info['fit_type']]
            # Recalculate the parameters
            fit_module.parameters = fit_module.get_params(self.info['fit_params'])

        self.upper_bound_inttext.observe(functools.partial(upper_bound_inttext_clicked, self=self))

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

        return ([self.add_range_button,self.adjust_range_button,self.fit_type_select,self.fit_button,self.clear_axis_button],
                    [self.add_range_button,self.adjust_range_button,self.fit_button,self.clear_axis_button],
                    [self.parameter_select,self.estimate_inttext,self.lower_bound_inttext,self.upper_bound_inttext,self.fit_order_inttext])

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
        'FitRange':[],
        'Fit':{'Axis':[]}
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
            if len(self.data[ax]['FitRange']) == 2: return

            # Clear all other points
            self.clear(self.artists[ax]['New Bound'],ax=ax,vlines=True)
            self.clear(self.artists[ax]['Selected Bound'],ax=ax,vlines=True)
            self.info['selected'] = False

            # Add the point to the array of boundaries
            self.data[ax]['FitRange'].append(event.xdata)

            # Replot all the points, including the extra one
            self.set_segments(self.artists[ax]['Bounds'],self.data[ax]['FitRange'],ax)

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
                hdists = self.hdist(event.xdata,self.data[ax]['FitRange'],ax)
                self.hdists = hdists
                # Find the index of the point closest to the click
                self.info['close_bound'] = np.nanargmin(hdists)

                # If the point is close to the click
                if hdists[self.info['close_bound']] < self.info['click_dist']:

                    # Replot the selected boundary in a different color
                    self.set_segments(self.artists[ax]['Selected Bound'],[self.data[ax]['FitRange'][self.info['close_bound']]],ax)
                    # State that a point has been selected
                    self.info['selected'] = True

            # If a point has already been selected
            else:
                # Remove the bound from the data array
                self.data[ax]['FitRange'] = np.delete(self.data[ax]['FitRange'],self.info['close_bound'])
                # Remove the temporary plotted point
                self.clear(self.artists[ax]['Selected Bound'],ax=ax,vlines=True)

                # Plot the new line
                self.set_segments(self.artists[ax]['New Bound'],[event.xdata],ax)
                # Add the point to the array of points
                self.data[ax]['FitRange'] = np.append(self.data[ax]['FitRange'],event.xdata)

                # Replot all points
                self.set_segments(self.artists[ax]['Bounds'],self.data[ax]['FitRange'],ax)

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
            if len(self.data[ax]['FitRange']) < 2: return

            if self.info['fit_type'] is not None:
                ## Import functions from a module
                imported_fit = self.info['fit_dict'][self.info['fit_type']]
                self.fit(imported_fit,ax,self.info,new_ax=imported_fit.new_ax)

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
        if self.show: plt.show()

    ##########################################
    ## ACTIVATE AND DEACTIVATE
    ##########################################

    # Overwrite the deactivation method called when the widget is turned off
    def activate(self):
        # Set the widget to the off setting
        self.widget_on = True
        # Display the widget menu
        self.place_menu(self.menu,self.button_list)
        # # Display the paramter menu
        # self.place_menu(self.param_menu,self.param_button_list)
        # Return the widget toggle
        return self.widget_on

    # Overwrite the deactivation method called when the widget is turned off
    def deactivate(self,main=False):
        # Deactivate the widget
        self.widget_on = False
        # Clear the widget menu
        self.place_menu(self.menu,[])
        # Clear the paramter menu
        self.place_menu(self.param_menu,[])
        # Return the widget toggle
        return self.widget_on

    ##########################################
    ## LOAD METHODS
    ##########################################

    def load(self,load_data):

        self.widget_on = True
        self.show = False

        class sim_event:
            def __init__(self,xdata,ydata,axis):
                self.xdata = xdata
                self.ydata = ydata
                self.inaxes = axis

        for axis in load_data['data']:
        # Here axis will take on values of 'Axis 1', 'Axis 2', etc.

            # Determine the axis on the new figure
            ax = self.fig.axes[int(axis.split(' ')[-1])-1]

            # Store the points which define the baseline
            if 'FitRange' in load_data['data'][axis]:
                range = load_data['data'][axis]['FitRange']
                self.info['interactive_mode'] = 'add_range'

                # Call the __call__ method to add the points with simulated events
                [self(sim_event(xdata,None,ax)) for xdata in range]

            else: continue

            # If there was a fit
            if 'Fit' in load_data['data'][axis]:

                fit = load_data['data'][axis]['Fit']

                # Loop through all the fits
                fit_num = 1
                while True:
                    if f'Fit {fit_num}' not in fit: break
                    else:
                        self.info['fit_type'] = fit[f'Fit {fit_num}']['fit_type']
                        self.info['fit_params'] = fit[f'Fit {fit_num}']['fit_params']
                        self.data[ax]['FitRange'] = fit[f'Fit {fit_num}']['FitRange']

                        # Apply fit order
                        self.info['fit_order'] = fit[f'Fit {fit_num}']['fit_order']
                        # Retrieve the fit module
                        fit_module = self.info['fit_dict'][self.info['fit_type']]
                        # Set the fit order in the module
                        fit_module.fit_order = self.info['fit_order']
                        # Recalculate the parameters
                        fit_module.parameters = fit_module.get_params(self.info['fit_params'])

                        self.info['interactive_mode'] = 'fit'

                        # Call the __call__ method to add the points with simulated events
                        self(sim_event(None,None,ax))

                        fit_num += 1

        self.widget_on = False
        self.show = True
        self.info['interactive_mode'] = 'off'

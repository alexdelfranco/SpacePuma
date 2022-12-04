# Author: Alex DelFranco
# Advisor: Rafa Martin Domenech
# Institution: Center for Astrophysics | Harvard & Smithsonian
# Date: 22 August 2022

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import functools

from scipy.interpolate import interp1d

from .base import widget_base

class baseline(widget_base):
    '''
    '''
    def __init__(self,fig,menu=None,artists_global=None,data_global=None,load_data=None):
        '''
        '''
        # Initialize the widget in the off position
        self.widget_on = False
        # Set the display setting to display output
        self.show = True
        # Setup the figure and axes
        self.fig = fig
        # Setup possible load data
        if load_data is not None:
            if 'baseline' in load_data: load_data = load_data['baseline']

        # Initialize all global matplotlib artists
        self.artists_global = self.pull_artists(artists_global)
        # Initialize all global data
        self.data_global = self.pull_data(data_global)
        # Create a local artist dictionary
        self.artists = {}

        # Create info and style dictionaries
        self.style,self.info,self.data = self.setup_defaults(load_data)

        # Initialize all buttons
        self.button_list,self.toggle_buttons = self.setup_buttons()
        # Place all the menu buttons
        if menu is None:
            self.menu = widgets.HBox()
            self.place_menu(self.menu,self.button_list)
        else: self.menu = menu

        if load_data is not None: self.load(load_data)

    # Define dictionaries of default values
    def setup_defaults(self,load_data=None):

        style = {
        'marker':'*',
        'markersize':15,
        'primary_color':'deepskyblue',
        'new_color':'red',
        'sel_color':'green',
        'fit_color':'darkviolet',
        'fit_color':'skyblue'}

        info = {
        # Initialize the selected boolean
        'selected':False,
        # Initialize the fit override order
        'fit_order_override':0,
        # Set the distance a click needs to be from a point to select it
        'click_dist':0.02,
        # Set a default fit type
        'fit_type':'Linear Segment',
        # Initialize the add, move, and delete mode holder
        'interactive_mode':'off',
        # Initialize an active axis
        'active_ax':None,
        # Initialize an array of axes with a baseline
        'baseline axes':[]
        }

        data = dict.fromkeys(self.artists_global['Interactive Axes'],self.data_init())

        if load_data is not None:
            load_data['info']['interactive_mode'] = 'off'
            load_data['info']['selected'] = False
            load_data['info']['active_ax'] = None

            return load_data['style'],load_data['info'],data

        return style,info,data

    # Define all buttons and their functionality
    def setup_buttons(self):

        ##########################################
        ## INTERACTIVE PLOTTING BUTTONS
        ##########################################

        # Add a button widget to add new points
        self.add_button = widgets.ToggleButton(description='Add Points')

        def on_add_button_clicked(b, self = self):

            # Turn off the other buttons
            for button in self.toggle_buttons:
                if button != self.add_button: button.value = False

            # Turn on add mode
            if self.info['interactive_mode'] == 'add':
                self.info['interactive_mode'] = 'off'
            else: self.info['interactive_mode'] = 'add'

        self.add_button.observe(functools.partial(on_add_button_clicked, self=self))

        #####################

        # Add a button widget to move points
        self.move_button = widgets.ToggleButton(description='Move Points')

        def on_move_button_clicked(b, self = self):

            # Turn off the other buttons
            for button in self.toggle_buttons:
                if button != self.move_button: button.value = False

            # Turn on move mode
            if self.info['interactive_mode'] == 'move':
                self.info['interactive_mode'] = 'off'
            else: self.info['interactive_mode'] = 'move'


        self.move_button.observe(functools.partial(on_move_button_clicked, self=self))

        #####################

        # Add a button widget to delete points
        self.delete_button = widgets.ToggleButton(description='Delete Points')

        def on_delete_button_clicked(b, self = self):

            # Turn off the other buttons
            for button in self.toggle_buttons:
                if button != self.delete_button: button.value = False

            # Turn on delete mode
            if self.info['interactive_mode'] == 'delete':
                self.info['interactive_mode'] = 'off'
            else: self.info['interactive_mode'] = 'delete'

        self.delete_button.observe(functools.partial(on_delete_button_clicked, self=self))

        ##########################################

        # Add a dropdown box to select the fit type
        self.fit_type_select = widgets.Select(description='Functional Form:',options=['Linear Segment','Polynomial'],rows=0,style={'description_width': 'initial'},layout = widgets.Layout(width='245px'))

        def on_fit_type_selected(b, self = self):

            # Change fit order value
            self.info['fit_type'] = self.fit_type_select.value
            if self.info['fit_type'] == 'Polynomial':
                self.button_list = [self.add_button,self.move_button,self.delete_button,self.fit_type_select,self.fit_order_inttext,self.clear_button]
                self.place_menu(self.menu,self.button_list)
            elif self.info['fit_type'] == 'Linear Segment':
                self.button_list = [self.add_button,self.move_button,self.delete_button,self.fit_type_select,self.clear_button]
                self.place_menu(self.menu,self.button_list)
            # Recalculate the fit
            for ax in self.info['baseline axes']:
                self.fit(ax)

        self.fit_type_select.observe(functools.partial(on_fit_type_selected, self=self))

        ##########################################

        # Add a text box to set the fit order
        self.fit_order_inttext = widgets.IntText(description='Order:',value=0,style={'description_width': 'initial'},layout = widgets.Layout(width='100px'))

        def on_fit_order_inttext_entered(b, self = self):

            # Change fit order value
            self.info['fit_order_override'] = self.fit_order_inttext.value
            # Recalculate the fit
            for ax in self.info['baseline axes']:
                self.fit(ax)

        self.fit_order_inttext.observe(functools.partial(on_fit_order_inttext_entered, self=self))

        #####################

        # Add a button widget to clear all points
        self.clear_button = widgets.ToggleButton(description='Clear Plot')

        def on_clear_button_clicked(b, self = self):

            # Turn off the other buttons
            for button in self.toggle_buttons:
                if button != self.clear_button: button.value = False

            # Turn on clear mode
            if self.info['interactive_mode'] == 'clear':
                self.info['interactive_mode'] = 'off'
            else: self.info['interactive_mode'] = 'clear'

        self.clear_button.observe(functools.partial(on_clear_button_clicked, self=self))

        #####################

        return [self.add_button,self.move_button,self.delete_button,self.fit_type_select,self.clear_button], \
        [self.add_button,self.move_button,self.delete_button,self.clear_button]

    ##########################################
    ## UPDATE METHODS
    ##########################################

    # Initialize Artists
    def new_artists(self,ax,style):

        # Plot
        bspoints, = ax.plot([],[],marker=style['marker'],linestyle='',color=style['primary_color'],markersize=style['markersize'])
        newpoint, = ax.plot([],[],color=style['new_color'],marker=style['marker'],markersize=style['markersize'])
        selpoint, = ax.plot([],[],color=style['sel_color'],marker=style['marker'],markersize=style['markersize'])
        plotfit, = ax.plot([],[],color=style['fit_color'])

        artists = {
        'Baseline Points': bspoints,
        'New Points': newpoint,
        'Selected Points': selpoint,
        'Baseline Func': plotfit
        }
        return artists

    def data_init(self):
        return {
            'Baseline':{
                # Fit points
                'Baseline Points': {'xdata':[],'ydata':[]},
                'Baseline': {'xdata':[],'ydata':[]},
                'Corrected Data': {'xdata':[],'ydata':[]},
                'Baseline Axis':None
                }
        }

    ##########################################
    ## EVENT HANDLER
    ##########################################

    # When the class is called as a function
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
            self.update_dictionaries(ax,artist_key='Baseline Points',data_key='Baseline')
            self.update(self.artists_global,self.data_global)

        # If a click is not within the axis, do nothing
        if self.info['active_ax'] == None: return
        else: ax = self.info['active_ax']

        ##########################################
        ## ADD POINTS
        ##########################################

        # If the add button has been pressed
        if self.info['interactive_mode'] == 'add':

            # Clear all other points
            self.clear(self.artists[ax]['New Points'])
            self.clear(self.artists[ax]['Selected Points'])
            self.info['selected'] = False

            # Add the point to the array of points
            self.data[ax]['Baseline']['Baseline Points']['xdata'] = np.append(self.data[ax]['Baseline']['Baseline Points']['xdata'],event.xdata)
            self.data[ax]['Baseline']['Baseline Points']['ydata'] = np.append(self.data[ax]['Baseline']['Baseline Points']['ydata'],event.ydata)

            # Replot all the points, including the extra one
            self.artists[ax]['Baseline Points'].set_data([self.data[ax]['Baseline']['Baseline Points']['xdata'],self.data[ax]['Baseline']['Baseline Points']['ydata']])

        ##########################################
        ## MOVE POINTS
        ##########################################

        # If the move button has been pressed
        elif self.info['interactive_mode'] == 'move':

            # If a point is not selected
            if not self.info['selected']:
                # If there was a newpoint added, clear it
                self.clear(self.artists[ax]['New Points'])
                # Calculate the distance from the click to each point
                euc = self.bigeuc(event.xdata,event.ydata,self.data[ax]['Baseline']['Baseline Points']['xdata'],self.data[ax]['Baseline']['Baseline Points']['ydata'],ax)
                # Find the index of the point closest to the click
                self.close = np.nanargmin(euc)

                # If the point is close to the click
                if euc[self.close] < self.info['click_dist']:

                    # Replot the selected point in a different color
                    self.artists[ax]['Selected Points'].set_data([self.data[ax]['Baseline']['Baseline Points']['xdata'][self.close],self.data[ax]['Baseline']['Baseline Points']['ydata'][self.close]])
                    # State that a point has been selected
                    self.info['selected'] = True

            # If a point has already been selected
            else:
                # Remove the point from the data arrays
                self.data[ax]['Baseline']['Baseline Points']['xdata'] = np.delete(self.data[ax]['Baseline']['Baseline Points']['xdata'],self.close)
                self.data[ax]['Baseline']['Baseline Points']['ydata'] = np.delete(self.data[ax]['Baseline']['Baseline Points']['ydata'],self.close)

                # Remove the temporary plotted point
                self.clear(self.artists[ax]['Selected Points'])

                # Plot the newly placed point
                self.artists[ax]['New Points'].set_data([event.xdata, event.ydata])

                # Add the point to the array of points
                self.data[ax]['Baseline']['Baseline Points']['xdata'] = np.append(self.data[ax]['Baseline']['Baseline Points']['xdata'],event.xdata)
                self.data[ax]['Baseline']['Baseline Points']['ydata'] = np.append(self.data[ax]['Baseline']['Baseline Points']['ydata'],event.ydata)

                # Replot all the points, excluding the selected one
                self.artists[ax]['Baseline Points'].set_data([self.data[ax]['Baseline']['Baseline Points']['xdata'],self.data[ax]['Baseline']['Baseline Points']['ydata']])

                # No point is currently selected
                self.info['selected'] = False

        ##########################################
        ## DELETE POINTS
        ##########################################

        # If the delete button has been pressed
        elif self.info['interactive_mode'] == 'delete':

            # Clear all other points
            self.clear(self.artists[ax]['New Points'])
            self.clear(self.artists[ax]['Selected Points'])
            self.info['selected'] = False

            # Calculate the distance from the click to each point
            euc = self.bigeuc(event.xdata,event.ydata,self.data[ax]['Baseline']['Baseline Points']['xdata'],self.data[ax]['Baseline']['Baseline Points']['ydata'],ax)
            # Find the index of the point closest to the click
            close = np.nanargmin(euc)

            # If the point is close to the click
            if euc[close] < self.info['click_dist']:
                # Remove the point from the data arrays
                self.data[ax]['Baseline']['Baseline Points']['xdata'] = np.delete(self.data[ax]['Baseline']['Baseline Points']['xdata'],close)
                self.data[ax]['Baseline']['Baseline Points']['ydata'] = np.delete(self.data[ax]['Baseline']['Baseline Points']['ydata'],close)

                # Replot all the points, excluding the selected one
                self.artists[ax]['Baseline Points'].set_data([self.data[ax]['Baseline']['Baseline Points']['xdata'],self.data[ax]['Baseline']['Baseline Points']['ydata']])

            # If there is only one point left, clear the fit
            if len(self.data[ax]['Baseline']['Baseline Points']['xdata']) < 2: self.clearfit()

        ##########################################
        ## CLEAR AXIS
        ##########################################

        # If the clear button has been pressed
        elif self.info['interactive_mode'] == 'clear':
            self.axis_clear(ax)

        ##########################################
        ## UPDATE ARTISTS
        ##########################################

        # If there is a fit, update it
        self.fit(ax)
        if self.show: plt.show()

    ##########################################
    ## FITTING METHODS
    ##########################################

    # Global fit function
    def fit(self,ax):
        # If there is only one point, don't fit
        if len(self.data[ax]['Baseline']['Baseline Points']['xdata']) < 2: return
        # Fit the baseline points depending using the specified function
        if self.info['fit_type'] == 'Polynomial': self.pfit(ax)
        elif self.info['fit_type'] == 'Linear Segment': self.lsfit(ax)
        # Setup baseline if not already setup
        if self.data[ax]['Baseline']['Baseline Axis'] is None:
            # Add axis to a list of axes with baselines
            if ax not in self.info['baseline axes']: self.info['baseline axes'].append(ax)
            # Add an axis to the figure
            self.resize_figure(len(self.fig.axes)+1,self.fig)
            # Save that axis
            baseline_ax = self.fig.axes[-1]
            self.data[ax]['Baseline']['Baseline Axis'] = baseline_ax
            # Add it to the interactive figure list
            self.artists_global['Interactive Axes'].append(baseline_ax)
            # Add a baselined plot
            self.artists[baseline_ax] = {}
            self.artists[baseline_ax]['Baselined Data'] = baseline_ax.plot([],[],color=self.style['primary_color'])[0]
            self.artists_global['Primary Artists'][baseline_ax] = 'Baselined Data'
        # # Update the baseline
        self.baseline_plot(ax)

    # Polynomial fit function
    def pfit(self,ax):
        # Set the fit order to one below the number of points
        self.fit_order = np.min([5,len(self.data[ax]['Baseline']['Baseline Points']['xdata'])-1])
        # If a fit order override is specified, override the fit order
        if self.info['fit_order_override'] != 0: self.fit_order = self.info['fit_order_override']
        # Calculate a polynomial fit
        z = np.polyfit(self.data[ax]['Baseline']['Baseline Points']['xdata'],self.data[ax]['Baseline']['Baseline Points']['ydata'], self.fit_order)
        self.fitfunc = np.poly1d(z)

        # Calculate the fit arrays
        xnew = np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],200)
        ynew = self.fitfunc(xnew)

        # Save the baseline
        self.data[ax]['Baseline']['Baseline']['xdata'] = xnew
        self.data[ax]['Baseline']['Baseline']['ydata'] = ynew
        # Plot the fit
        self.artists[ax]['Baseline Func'].set_data([xnew,ynew])

        # Make sure the fit degree is visually up to date
        self.fit_order_inttext.value = self.fit_order

    # Linear segmented fit function
    def lsfit(self,ax):
        # Calculate ordered copies of xdat and ydat
        xord,yord = self.order(self.data[ax]['Baseline']['Baseline Points']['xdata'],self.data[ax]['Baseline']['Baseline Points']['ydata'])
        # Create the fitting function
        self.fitfunc = interp1d(xord,yord)

        # Get the length of the data array
        bsx,_ = self.artists_global[ax][self.artists_global['Primary Artists'][ax]].get_data()
        dat_length = len(np.squeeze(np.where((bsx > xord[0]) & (bsx < xord[-1]))))
        # Create x and y data using the fitting function
        xnew = np.linspace(np.min(xord),np.max(xord),dat_length)
        ynew = self.fitfunc(xnew)

        # Save the baseline
        self.data[ax]['Baseline']['Baseline']['xdata'] = xnew
        self.data[ax]['Baseline']['Baseline']['ydata'] = ynew
        # Plot the fit
        self.artists[ax]['Baseline Func'].set_data([xnew,ynew])

    # Baseline method
    def baseline_plot(self,ax):
        # Subset the x range to what is visible
        xord,yord = self.order(self.data[ax]['Baseline']['Baseline Points']['xdata'],self.data[ax]['Baseline']['Baseline Points']['ydata'])
        # Subset the science data with the range defined by the fit
        bsx,bsy = self.artists_global[ax][self.artists_global['Primary Artists'][ax]].get_data()
        ii = np.squeeze(np.where((bsx > xord[0]) & (bsx < xord[-1])))
        # Redefine the baseline vars as their relevant subsets
        bsx,bsy = bsx[ii],bsy[ii]
        # Calculate the fit using the baseline x data
        bs_fity = self.fitfunc(bsx)
        # Save the baseline-subtracted data
        self.data[ax]['Baseline']['Corrected Data']['xdata'] = bsx
        self.data[ax]['Baseline']['Corrected Data']['ydata'] = bsy - bs_fity
        # Show the residual plot
        self.artists[self.data[ax]['Baseline']['Baseline Axis']]['Baselined Data'].set_data([bsx, bsy - bs_fity])
        # Autoscale the y axis correspondingly
        self.data[ax]['Baseline']['Baseline Axis'].relim()
        self.data[ax]['Baseline']['Baseline Axis'].autoscale()

    def axis_clear(self,ax):
        # If there is a baseline axis that was generated from this one
        if self.data[ax]['Baseline']['Baseline Axis'] is not None:
            # Recursively clear it
            self.axis_clear(self.data[ax]['Baseline']['Baseline Axis'])

            # Clear baseline axis entry in artists dict
            del self.artists[self.data[ax]['Baseline']['Baseline Axis']]
            # Clear baseline axis entry in artists dict
            del self.data[self.data[ax]['Baseline']['Baseline Axis']]
            # Clear baseline axis entry in global artists dict
            if self.data[ax]['Baseline']['Baseline Axis'] in self.artists_global.keys():
                del self.artists_global[self.data[ax]['Baseline']['Baseline Axis']]
                self.artists_global['Interactive Axes'].remove(self.data[ax]['Baseline']['Baseline Axis'])
            # Clear baseline axis entry in global artists dict
            if self.data[ax]['Baseline']['Baseline Axis'] in self.data_global.keys():
                del self.data_global[self.data[ax]['Baseline']['Baseline Axis']]

            # Remove the baseline axis
            self.fig.delaxes(self.data[ax]['Baseline']['Baseline Axis'])
            del self.artists_global['Primary Artists'][self.data[ax]['Baseline']['Baseline Axis']]
            self.data[ax]['Baseline']['Baseline Axis'] = None
            self.resize_figure(len(self.fig.axes),self.fig)

        # Clear specified artists
        for artist in self.artists[ax]:
            if artist != 'Baselined Data':
                self.clear(self.artists[ax][artist])

        # Reset selected boolean
        self.info['selected'] = False

        # Clear data dictionary
        self.data[ax] = self.data_init()

    ##########################################
    ## LOAD METHODS
    ##########################################

    def load(self,load_data):

        self.widget_on = True
        self.show = False
        self.info['interactive_mode'] = 'add'

        class sim_event:
            def __init__(self,xdata,ydata,axis):
                self.xdata = xdata
                self.ydata = ydata
                self.inaxes = axis

        for axnum,axis in enumerate(load_data['data']):
        # Here axis will take on values of 'Axis 1', 'Axis 2', etc.

            # If new axes have yet to be added, return
            if axnum >= len(self.fig.axes): break

            # Determine the axis on the new figure
            ax = self.fig.axes[int(axis.split(' ')[-1])-1]

            # Store the points which define the baseline
            if 'Baseline' in load_data['data'][axis]:
                baseline_pts = load_data['data'][axis]['Baseline']['Baseline Points']

                # Call the baseline __call__ method to add the points with simulated events
                [self(sim_event(xdata,ydata,ax)) for xdata,ydata in zip(baseline_pts['xdata'],baseline_pts['ydata'])]

        self.widget_on = False
        self.show = True
        self.info['interactive_mode'] = 'off'

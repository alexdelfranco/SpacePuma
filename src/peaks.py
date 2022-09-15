# Author: Alex DelFranco
# Advisor: Rafa Martin Domenech
# Institution: Center for Astrophysics | Harvard & Smithsonian
# Date: 14 September 2022

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import functools

from scipy.signal import find_peaks,medfilt
import seaborn as sns

from base import widget_base

class peaks(widget_base):
    def __init__(self,fig,menu=None,data=None,artists_global=None,data_global=None):
        '''
        '''
        self.widget_on = False
        # Setup the figure
        self.fig = fig

        # Initialize all global matplotlib artists
        self.artists_global = self.pull_artists(artists_global)
        # Initialize all global data
        self.data_global = self.pull_data(data_global)
        # Create a dictionary using the axes as keys
        self.artists = {}

        # Initialize defaults
        self.style,self.info,self.data = self.setup_defaults()

        # Initialize all buttons
        self.button_list,self.toggle_buttons = self.setup_buttons()
        # Place all the menu buttons
        if menu is None:
            self.menu = widgets.HBox()
            self.place_menu(self.menu,self.button_list)
        else: self.menu = menu

    def setup_defaults(self):
        style = {
        'fill_color':'deepskyblue',
        'alpha':0.6,
        'vline_style':'--',
        'vline_width':2,
        'color_palette':'Set2',
        'peak_style':'-',
        'peak_width':1
        }

        info = {
        # Initialize the add, move, and delete
        'interactive_mode':'off',
        # Initialize an active axis
        'active_ax':None,
        # Initialize a selected boolean
        'selected':False,
        # Set an interactive click distance
        'click_dist':0.02,
        # Initialize a height
        'height':None,
        # Initialize a distance
        'distance':None
        }

        data = dict.fromkeys(self.artists_global['Interactive Axes'],self.data_init())

        return style,info,data

    def data_init(self):
        return {
        'Range':[],
        'Peaks':{'xdata':[],'Display':False},
        }

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

        # Add a button widget to add new points
        self.adjust_range_button = widgets.ToggleButton(description='Adjust Range')

        def on_add_range_button_clicked(b, self = self):

            # Turn off the other buttons
            for button in self.toggle_buttons:
                if button != self.adjust_range_button: button.value = False

            # Turn on adjust_range mode
            if self.info['interactive_mode'] == 'adjust_range':
                self.info['interactive_mode'] = 'off'
            else: self.info['interactive_mode'] = 'adjust_range'

        self.adjust_range_button.observe(functools.partial(on_add_range_button_clicked, self=self))

        #####################

        # Add a button widget to add new points
        self.find_peaks_button = widgets.ToggleButton(description='Find Peaks')

        def find_peaks_button_clicked(b, self = self):

            # Turn off the other buttons
            for button in self.toggle_buttons:
                if button != self.find_peaks_button: button.value = False

            # Turn on integrate mode
            if self.info['interactive_mode'] == 'find_peaks':
                self.info['interactive_mode'] = 'off'
            else: self.info['interactive_mode'] = 'find_peaks'

        self.find_peaks_button.observe(functools.partial(find_peaks_button_clicked, self=self))

        #####################

        # Add a button widget to add new points
        self.height_floattext = widgets.FloatText(description='Height:',value=0,style={'description_width':'initial'},layout = widgets.Layout(width='115px'))

        def height_floattext_clicked(b, self = self):

            if self.distance_floattext.value != 0:
                self.info['height'] = self.distance_floattext.value
            else: self.info['height'] = None

        self.height_floattext.observe(functools.partial(height_floattext_clicked, self=self))

        #####################

        # Add a button widget to add new points
        self.distance_floattext = widgets.FloatText(description='Distance:',value=0,style={'description_width':'initial'},layout = widgets.Layout(width='130px'))

        def distance_floattext_clicked(b, self = self):

            if self.distance_floattext.value != 0:
                self.info['distance'] = self.distance_floattext.value
            else: self.info['distance'] = None

        self.distance_floattext.observe(functools.partial(distance_floattext_clicked, self=self))

        #####################

        # Add a button widget to add new points
        self.clear_axis_button = widgets.ToggleButton(description='Clear Axis')

        def clear_axis_button_clicked(b, self = self):

            # Turn off the other buttons
            for button in self.toggle_buttons:
                if button != self.clear_axis_button: button.value = False

            # Turn on integrate mode
            if self.info['interactive_mode'] == 'clear':
                self.info['interactive_mode'] = 'off'
            else: self.info['interactive_mode'] = 'clear'

        self.clear_axis_button.observe(functools.partial(clear_axis_button_clicked, self=self))

        #####################

        return [self.add_range_button,self.adjust_range_button,self.find_peaks_button,self.height_floattext,self.distance_floattext,self.clear_axis_button],[self.add_range_button,self.adjust_range_button,self.find_peaks_button,self.clear_axis_button]

    # Method to create a new artist list
    def new_artists(self,ax,style):
        # Find the y axis minimum and maximum
        ymin,ymax = ax.get_ylim()

        artists = {
        'Bounds': ax.vlines([],ymin,ymax,linestyle=self.style['vline_style'],linewidth=self.style['vline_width']),
        'New Bound': ax.vlines([],ymin,ymax,linestyle=self.style['vline_style'],linewidth=self.style['vline_width'],color='red'),
        'Selected Bound': ax.vlines([],ymin,ymax,linestyle=self.style['vline_style'],linewidth=self.style['vline_width'],color='green'),
        'Peaks': ax.vlines([],ymin,ymax,linestyle=self.style['peak_style'],linewidth=self.style['peak_width'],color='red')
        }
        return artists

    def __call__(self,event):

        if not self.widget_on: return
        self.info['active_ax'] = None

        ##########################################
        ## UPDATE ARTISTS
        ##########################################

        for ax in self.fig.axes:
            self.test = 2
            # Determine which axis was clicked on
            if event.inaxes == ax: self.info['active_ax'] = ax
            # If the axis is not interactive, do not add new artists
            if ax not in self.artists_global['Interactive Axes']: continue
            # If the artist is interactive, initialize the appropriate artists
            self.update_dictionaries(ax,artist_key='Bounds',data_key='Peaks')
            self.update(self.artists_global,self.data_global)

        self.test = 'e'
        # If a click is not within the axis, do nothing
        if self.info['active_ax'] == None: return
        else: ax = self.info['active_ax']

        ##########################################
        ## ADD RANGE
        ##########################################

        self.test = 'c'
        # If in add range mode
        if self.info['interactive_mode'] == 'add_range':

            # Check to see if the axis is an interactive axis
            if ax not in self.artists_global['Interactive Axes']: return
            # Check to see how many bounds are already in the axis
            if len(self.data[ax]['Range']) == 2: return

            self.test = 'b'
            # Clear all other points
            self.clear(self.artists[ax]['New Bound'],ax=ax,vlines=True)
            self.clear(self.artists[ax]['Selected Bound'],ax=ax,vlines=True)
            self.info['selected'] = False

            self.test = 'a'
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
        ## FIND PEAKS
        ##########################################

        # If in find peaks mode
        elif self.info['interactive_mode'] == 'find_peaks':

            # Check to see if the axis is an interactive axis
            if ax not in self.artists_global['Interactive Axes']: return

            if not self.data[ax]['Peaks']['Display']:
                # Plot the peaks
                self.peaks(ax)
                # Set display to true
                self.data[ax]['Peaks']['Display'] = True
            else:
                # Clear the peaks
                self.set_segments(self.artists[ax]['Peaks'],[],ax)
                # Set display to false
                self.data[ax]['Peaks']['Display'] = False

        ##########################################
        ## CLEAR AXIS
        ##########################################

        # If in clear mode
        elif self.info['interactive_mode'] == 'clear':

            # If the artist is not meant for data display
            if ax in self.artists_global['Interactive Axes']:

                # Clear the data dictionary
                self.data[ax] = self.data_init()
                # Clear artists
                self.set_segments(self.artists[ax]['Bounds'],[],ax)
                self.set_segments(self.artists[ax]['New Bound'],[],ax)
                self.set_segments(self.artists[ax]['Selected Bound'],[],ax)

                # Clear the peaks
                self.set_segments(self.artists[ax]['Peaks'],[],ax)

    ##########################################
    ## PEAKS METHODS
    ##########################################

    def peaks(self,ax):

        # Set the data
        height = self.data[ax]['height'] = self.info['height']
        dist = self.data[ax]['distance'] = self.info['distance']

        # Determine the primary curve for the current axis
        curve = self.artists_global['Primary Artists'][ax]
        xdata,ydata = self.artists_global[ax][curve].get_data()

        # Find the indeces of the data that are within the range
        xmin,xmax = sorted(self.data[ax]['Range'])
        ii = np.squeeze(np.where((xdata > xmin) & (xdata < xmax)))

        # Subset the x and y data
        xsub,ysub = xdata[ii],ydata[ii]

        # Find the peak indices
        ii = self.peak_vals(ysub,height,dist)

        # Save the xvalues of the peaks to the data dictionary
        self.data[ax]['Peaks']['xdata'] = xsub[ii]
        self.set_segments(self.artists[ax]['Peaks'],self.data[ax]['Peaks']['xdata'],ax)

    def peak_vals(self,spec,height,distance):
        normspec = spec/np.nanmax(spec)
        sp_medavg = medfilt(normspec,kernel_size=51)
        flat_spec = np.abs(normspec - sp_medavg)
        peaks,_ = find_peaks(flat_spec,height=height,distance=distance)
        return peaks

    ##########################################
    ## UPDATE METHODS
    ##########################################

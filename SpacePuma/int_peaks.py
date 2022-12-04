# Author: Alex DelFranco
# Advisor: Rafa Martin Domenech
# Institution: Center for Astrophysics | Harvard & Smithsonian
# Date: 14 September 2022

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import functools

from scipy.signal import find_peaks,medfilt
from scipy.integrate import simps
import seaborn as sns
import warnings

import logging
logging.getLogger().setLevel(logging.CRITICAL)

warnings.filterwarnings("ignore", message="kernel_size exceeds volume extent: the volume will be zero-padded.")

from .base import widget_base

class int_peaks(widget_base):
    def __init__(self,fig,menu=None,data=None,artists_global=None,data_global=None,load_data=None):
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
            if 'int_peaks' in load_data: load_data = load_data['int_peaks']

        # Initialize all global matplotlib artists
        self.artists_global = self.pull_artists(artists_global)
        # Initialize all global data
        self.data_global = self.pull_data(data_global)
        # Create a dictionary using the axes as keys
        self.artists = {}

        # Initialize defaults
        self.style,self.info,self.data = self.setup_defaults(load_data)

        # Initialize all buttons
        self.button_list,self.toggle_buttons = self.setup_buttons()
        # Place all the menu buttons
        if menu is None:
            self.menu = widgets.HBox()
            self.place_menu(self.menu,self.button_list)
        else: self.menu = menu

        if load_data is not None: self.load(load_data)

    def setup_defaults(self,load_data=None):

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

        if load_data is not None:
            load_data['info']['interactive_mode'] = 'off'
            load_data['info']['selected'] = False
            load_data['info']['active_ax'] = None

            return load_data['style'],load_data['info'],data

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

        # Add a button widget to add new points
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

        # Add a button widget to add new points
        self.integrate_button = widgets.ToggleButton(description='Integrate')

        def integrate_button_clicked(b, self = self):

            # Turn off the other buttons
            for button in self.toggle_buttons:
                if button != self.integrate_button: button.value = False

            # Turn on integrate mode
            if self.info['interactive_mode'] == 'integrate':
                self.info['interactive_mode'] = 'off'
            else: self.info['interactive_mode'] = 'integrate'

        self.integrate_button.observe(functools.partial(integrate_button_clicked, self=self))

        #####################

        # Add a button widget to add new points
        self.find_peaks_button = widgets.ToggleButton(description='Find Peaks')

        def find_peaks_button_clicked(b, self = self):

            # Turn off the other buttons
            for button in self.toggle_buttons:
                if button != self.find_peaks_button: button.value = False

            # Turn on find peaks mode
            if self.info['interactive_mode'] == 'find_peaks':
                self.info['interactive_mode'] = 'off'
            else: self.info['interactive_mode'] = 'find_peaks'

        self.find_peaks_button.observe(functools.partial(find_peaks_button_clicked, self=self))

        #####################

        # Add a button widget to add new points
        self.height_floattext = widgets.FloatText(description='Height:',value=0,style={'description_width':'initial'},layout = widgets.Layout(width='115px'))

        def height_floattext_clicked(b, self = self):

            if self.height_floattext.value != 0:
                self.info['height'] = self.height_floattext.value
            else: self.info['height'] = None

        self.height_floattext.observe(functools.partial(height_floattext_clicked, self=self))

        #####################

        # Add a button widget to add new points
        self.distance_floattext = widgets.FloatText(description='Distance:',value=0,style={'description_width':'initial'},layout=widgets.Layout(width='130px'))

        def distance_floattext_clicked(b, self = self):

            if self.distance_floattext.value != 0:
                self.info['distance'] = self.distance_floattext.value
            else: self.info['distance'] = None

        self.distance_floattext.observe(functools.partial(distance_floattext_clicked, self=self))

        #####################

        # Add a button widget to add new points
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

        return [self.add_range_button,self.adjust_range_button,self.integrate_button,self.find_peaks_button,self.height_floattext,self.distance_floattext,self.clear_axis_button],[self.add_range_button,self.adjust_range_button,self.integrate_button,self.find_peaks_button,self.clear_axis_button]

    ##########################################
    ## UPDATE METHODS
    ##########################################

    # Method to create a new artist list
    def new_artists(self,ax,style):
        # Find the y axis minimum and maximum
        ymin,ymax = ax.get_ylim()

        artists = {
        'Bounds': ax.vlines([],ymin,ymax,linestyle=self.style['vline_style'],linewidth=self.style['vline_width']),
        'New Bound': ax.vlines([],ymin,ymax,linestyle=self.style['vline_style'],linewidth=self.style['vline_width'],color='red'),
        'Selected Bound': ax.vlines([],ymin,ymax,linestyle=self.style['vline_style'],linewidth=self.style['vline_width'],color='green'),
        'Peaks': ax.vlines([],ymin,ymax,linestyle=self.style['peak_style'],linewidth=self.style['peak_width'],color='red'),
        'Integration': ax.fill_between([],[])
        }
        return artists

    def data_init(self):
        return {
        'IntPeakRange':[],
        'Integration':{'xdata':[],'ydata':[],'Display':False},
        'Peaks':{'xdata':[],'Display':False},
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
            self.update_dictionaries(ax,artist_key='Bounds',data_key='Peaks')
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
            if len(self.data[ax]['IntPeakRange']) == 2: return

            # Clear all other points
            self.clear(self.artists[ax]['New Bound'],ax=ax,vlines=True)
            self.clear(self.artists[ax]['Selected Bound'],ax=ax,vlines=True)
            self.info['selected'] = False

            # Add the point to the array of boundaries
            self.data[ax]['IntPeakRange'].append(event.xdata)

            # Replot all the points, including the extra one
            self.set_segments(self.artists[ax]['Bounds'],self.data[ax]['IntPeakRange'],ax)

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
                hdists = self.hdist(event.xdata,self.data[ax]['IntPeakRange'],ax)
                self.hdists = hdists
                # Find the index of the point closest to the click
                self.info['close_bound'] = np.nanargmin(hdists)

                # If the point is close to the click
                if hdists[self.info['close_bound']] < self.info['click_dist']:

                    # Replot the selected boundary in a different color
                    self.set_segments(self.artists[ax]['Selected Bound'],[self.data[ax]['IntPeakRange'][self.info['close_bound']]],ax)
                    # State that a point has been selected
                    self.info['selected'] = True

            # If a point has already been selected
            else:
                # Remove the bound from the data array
                self.data[ax]['IntPeakRange'] = np.delete(self.data[ax]['IntPeakRange'],self.info['close_bound'])
                # Remove the temporary plotted point
                self.clear(self.artists[ax]['Selected Bound'],ax=ax,vlines=True)

                # Plot the new line
                self.set_segments(self.artists[ax]['New Bound'],[event.xdata],ax)
                # Add the point to the array of points
                self.data[ax]['IntPeakRange'] = np.append(self.data[ax]['IntPeakRange'],event.xdata)

                # Replot all points
                self.set_segments(self.artists[ax]['Bounds'],self.data[ax]['IntPeakRange'],ax)

                # Replot the integration
                if self.data[ax]['Integration']['Display']: self.integrate(ax)

                # No point is currently selected
                self.info['selected'] = False

        ##########################################
        ## INTEGRATE
        ##########################################

        # If in integrate mode
        elif self.info['interactive_mode'] == 'integrate':

            # Check to see if the axis is an interactive axis
            if ax not in self.artists_global['Interactive Axes']: return

            if not self.data[ax]['Integration']['Display']:
                # Plot the peaks
                self.integrate(ax)
                # Set display to true
                self.data[ax]['Integration']['Display'] = True
            else:
                # Clear the peaks
                self.artists[ax]['Integration'].set_paths([])
                # Remove the peak from the legend
                self.artists[ax]['Integration'].set_label('_')
                ax.legend()
                # Set display to false
                self.data[ax]['Integration']['Display'] = False

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

                # Clear the integration
                self.artists[ax]['Integration'].set_paths([])\

                # Clear the peaks
                self.set_segments(self.artists[ax]['Peaks'],[],ax)

                # Clear the legend
                ax.get_legend().remove()

        ##########################################
        ## UPDATE ARTISTS
        ##########################################

        # Update artists
        if self.show: plt.show()

    ##########################################
    ## INTEGRATION METHODS
    ##########################################

    def integrate(self,ax):

        # Display the integration
        self.data[ax]['Integration']['Display'] = True

        # Clear the past integration
        self.artists[ax]['Integration'].set_paths([])

        # Determine the primary curve for the current axis
        curve = self.artists_global['Primary Artists'][ax]
        # Pull out the x and y data from that curve
        xdata,ydata = self.artists_global[ax][curve].get_data()
        # Determine the range of integration
        xmin,xmax = sorted(self.data[ax]['IntPeakRange'])

        # Determine the indices of the data values within the range
        ii = np.squeeze(np.where((xdata > xmin) & (xdata < xmax)))
        # Redefine the baseline vars as their relevant subsets
        xdata,ydata = xdata[ii],ydata[ii]

        self.data[ax]['Integration']['xdata'] = xdata
        self.data[ax]['Integration']['ydata'] = ydata

        # Use scipy's simpsons integration technique -- negative for decreasing x range
        self.data[ax]['Integration']['area'] = area = simps(y=ydata,x=xdata)

        self.artists[ax]['Integration'].set_label('_')

        # Plot the integration
        self.artists[ax]['Integration'] = ax.fill_between(xdata,ydata,label=f'Area: {round(area,3)}',color='dodgerblue',alpha=0.6)
        ax.legend()

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
        xmin,xmax = sorted(self.data[ax]['IntPeakRange'])
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

        for axnum,axis in enumerate(load_data['data']):
        # Here axis will take on values of 'Axis 1', 'Axis 2', etc.

            # If new axes have yet to be added, return
            if axnum >= len(self.fig.axes): break

            # Determine the axis on the new figure
            ax = self.fig.axes[int(axis.split(' ')[-1])-1]

            # Store the points which define the baseline
            if 'IntPeakRange' in load_data['data'][axis]:
                range = load_data['data'][axis]['IntPeakRange']
                self.info['interactive_mode'] = 'add_range'

                # Call the __call__ method to add the points with simulated events
                [self(sim_event(xdata,None,ax)) for xdata in range]

            else: continue

            # If there was an integration or a series of peaks
            if 'Integration' in load_data['data'][axis]:

                # Determine whether the integration should be displayed
                if load_data['data'][axis]['Integration']['Display']:

                    self.info['interactive_mode'] = 'integrate'
                    # Call the __call__ method to add the points with simulated events
                    self(sim_event(None,None,ax))

            if 'Peaks' in load_data['data'][axis]:

                # Determine whether the peaks should be displayed
                if load_data['data'][axis]['Peaks']['Display']:

                    self.info['interactive_mode'] = 'find_peaks'
                    # Call the __call__ method to add the points with simulated events
                    self(sim_event(None,None,ax))

        self.widget_on = False
        self.show = True
        self.info['interactive_mode'] = 'off'

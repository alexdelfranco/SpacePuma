# Author: Alex DelFranco
# Advisor: Rafa Martin Domenech
# Institution: Center for Astrophysics | Harvard & Smithsonian
# Date: 23 September 2022

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as widgets
import functools

from .base import widget_base

class display_tools(widget_base):
    def __init__(self,fig,menu=None,artists_global=None):
        '''
        '''
        self.widget_on = False
        self.fig = fig

        # Initialize all global matplotlib artists
        self.artists_global = self.pull_artists(artists_global)

        # Initialize defaults
        self.info = self.setup_defaults()

        # Initialize all buttons
        self.button_list,self.toggle_buttons = self.setup_buttons()
        # Place all the menu buttons
        if menu is None:
            self.menu = widgets.HBox()
            self.place_menu(self.menu,self.button_list)
        else: self.menu = menu

    def setup_defaults(self):
        info = {
        'axis':self.fig.axes[0],
        'property':'',
        'property_value':None,
        'artist':self.artists_global[self.fig.axes[0]]['Input Data'],
        'interactive_mode':'off',
        'active_ax':None,
        'scale':'linear'
        }

        return info

    def setup_buttons(self):

        ##########################################
        ## INTERACTIVE PLOTTING BUTTONS
        ##########################################

        # Add a button widget to add new points
        self.axis_select = widgets.Select(description='Axis:',options=range(1,len(self.fig.axes)+1),rows=0,style={'description_width': 'initial'},layout = widgets.Layout(width='95px'))

        def on_axis_selected(b, self = self):

            self.info['axis'] = self.fig.axes[self.axis_select.value-1]
            self.artist_select.options = list(self.artists_global[self.info['axis']].keys())

        self.axis_select.observe(functools.partial(on_axis_selected, self=self))

        #####################

        # Add a button widget to add new points
        self.artist_select = widgets.Select(description='Artist:',options=list(self.artists_global[self.info['axis']].keys()),rows=0,style={'description_width': 'initial'},layout = widgets.Layout(width='210px'))

        def on_artist_selected(b, self = self):

            if self.artist_select.value in self.artists_global[self.info['axis']].keys():
                self.info['artist'] = self.artists_global[self.info['axis']][self.artist_select.value]
                artist = self.info['artist']
            else: return

            # If the artist is a plot
            if isinstance(artist,matplotlib.lines.Line2D):
                self.property_select.options = ['Color','Marker','Markersize','Style','Width','Alpha']
            # If the artist is a vlines collections
            elif isinstance(artist,matplotlib.collections.LineCollection):
                self.property_select.options = ['Color','Style','Width','Alpha']
            # If the artist is a fill_between
            elif isinstance(artist,matplotlib.collections.PolyCollection):
                self.property_select.options = ['Color','Alpha']

        self.artist_select.observe(functools.partial(on_artist_selected, self=self))

        #####################

        # Add a button widget to add new points
        self.property_select = widgets.Select(description='Property:',options=['Color','Marker','Markersize','Style','Width','Alpha'],rows=0,style={'description_width': 'initial'},layout = widgets.Layout(width='180px'))

        def on_property_selected(b, self = self):

            # Change saved property value
            self.info['property'] = self.property_select.value
            artist = self.info['artist']

            if self.info['property'] == 'Color':
                self.button_list = [self.axis_select,self.artist_select,self.property_select,self.color_picker,self.scale_button,self.scale_select]
                self.place_menu(self.menu,self.button_list)
                if isinstance(artist,matplotlib.collections.PolyCollection) or isinstance(artist,matplotlib.collections.LineCollection):
                    color = artist.get_facecolor()
                    hexcolor = '#{:02x}{:02x}{:02x}'.format(*[int(n) for n in color[0][:3]*255])
                    self.color_picker.value = hexcolor
                else: self.color_picker.value = artist.get_color()
            elif self.info['property'] == 'Marker':
                self.button_list = [self.axis_select,self.artist_select,self.property_select,self.prop_text,self.scale_button,self.scale_select]
                self.place_menu(self.menu,self.button_list)
                self.prop_text.value = artist.get_marker()
                self.prop_text.description = 'Marker:'
            elif self.info['property'] == 'Markersize':
                self.button_list = [self.axis_select,self.artist_select,self.property_select,self.prop_float,self.scale_button,self.scale_select]
                self.place_menu(self.menu,self.button_list)
                self.prop_float.value = artist.get_markersize()
                self.prop_float.description = 'Marker Size:'
            elif self.info['property'] == 'Style':
                self.button_list = [self.axis_select,self.artist_select,self.property_select,self.prop_text,self.scale_button,self.scale_select]
                self.place_menu(self.menu,self.button_list)
                self.prop_text.value = artist.get_linestyle()
                self.prop_text.description = 'Style:'
            elif self.info['property'] == 'Width':
                self.button_list = [self.axis_select,self.artist_select,self.property_select,self.prop_float,self.scale_button,self.scale_select]
                self.place_menu(self.menu,self.button_list)
                self.prop_float.value = artist.get_linewidth()
                self.prop_float.description = 'Width:'
            elif self.info['property'] == 'Alpha':
                self.button_list = [self.axis_select,self.artist_select,self.property_select,self.prop_float,self.scale_button,self.scale_select]
                self.place_menu(self.menu,self.button_list)
                if artist.get_alpha() is None: self.prop_float.value = 1
                else: self.prop_float.value = artist.get_alpha()
                self.prop_float.description = 'Alpha:'

        self.property_select.observe(functools.partial(on_property_selected, self=self))

        #####################

        # Add a button widget to add new points
        self.color_picker = widgets.ColorPicker(description='Color:',value='deepskyblue',concise=True,style={'description_width': 'initial'},layout = widgets.Layout(width='90px'))

        def on_color_picked(b, self = self):

            artist = self.info['artist']

            artist.set(color=self.color_picker.value)

            plt.show()

        self.color_picker.observe(functools.partial(on_color_picked, self=self))

        #####################

        # Add a button widget to add new points
        self.prop_text = widgets.Text(description='Text:',value='',style={'description_width': 'initial'},layout=widgets.Layout(width='150px'))

        def on_prop_text_entered(b, self = self):

            artist = self.info['artist']

            if self.info['property'] == 'Style':
                if self.prop_text.value not in ['-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted']:
                    artist.set(linestyle='-')
                    return
                artist.set(linestyle=self.prop_text.value)
            if self.info['property'] == 'Marker':
                if self.prop_text.value not in np.concatenate(list(matplotlib.lines.Line2D.markers.items())): return
                artist.set(marker=self.prop_text.value)

            plt.show()

        self.prop_text.observe(functools.partial(on_prop_text_entered, self=self))

        #####################

        # Add a button widget to add new points
        self.prop_float = widgets.FloatText(description='Float:',value=1,style={'description_width': 'initial'},layout=widgets.Layout(width='150px'))

        def on_prop_float_entered(b, self = self):

            artist = self.info['artist']

            if self.info['property'] == 'Markersize':
                artist.set(markersize=self.prop_float.value)
            if self.info['property'] == 'Width':
                artist.set(linewidth=self.prop_float.value)
            if self.info['property'] == 'Alpha':
                if self.prop_float.value < 0 or self.prop_float.value > 1:
                    artist.set(alpha=1)
                    return
                artist.set(alpha=self.prop_float.value)

            plt.show()

        self.prop_float.observe(functools.partial(on_prop_float_entered, self=self))

        #####################

        # Add a button widget to save the figure
        self.scale_button = widgets.ToggleButton(description='Scale Plot')

        def on_scale_button_clicked(b, self = self):

            # Turn on add_range mode
            if self.info['interactive_mode'] == 'scale':
                self.info['interactive_mode'] = 'off'
            else: self.info['interactive_mode'] = 'scale'

        self.scale_button.observe(functools.partial(on_scale_button_clicked, self=self))

        #####################

        # Add a button widget to add new points
        self.scale_select = widgets.Select(description='Scale:',options=['linear','log','symlog','logit'],rows=0,style={'description_width': 'initial'},layout = widgets.Layout(width='140px'))

        def on_scale_selected(b, self = self):

            self.info['scale'] = self.scale_select.value

        self.scale_select.observe(functools.partial(on_scale_selected, self=self))

        #####################

        return [self.axis_select,self.artist_select,self.property_select,self.color_picker,self.scale_button,self.scale_select],[self.scale_button]

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

        # If a click is not within the axis, do nothing
        if self.info['active_ax'] == None: return
        else: ax = self.info['active_ax']

        ##########################################
        ## ADD RANGE
        ##########################################

        # If in add range mode
        if self.info['interactive_mode'] == 'scale':
            ax.set_yscale(self.info['scale'])

        plt.show()

    def update(self,artists,data):
        self.artists_global = artists
        self.axis_select.options = range(1,len(self.fig.axes)+1)
        self.artist_select.options = list(self.artists_global[self.info['axis']].keys())

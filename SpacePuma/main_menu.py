# Author: Alex DelFranco
# Advisor: Rafa Martin Domenech
# Institution: Center for Astrophysics | Harvard & Smithsonian
# Date: 22 August 2022

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import ipywidgets as widgets
import copy as cp
import pickle as pkl

from .base import widget_base
from .baseline import baseline
from .fit import fit
from .int_peaks import int_peaks
from .labels import labels
from .display_tools import display_tools
from .export import export

from scipy import integrate as scpyintegrate
import functools

class main_menu(widget_base):
    def __init__(self,data,load_file=None,fig=None,ax=None,exp_path=''):
        '''
        Initializes a widget instance
        '''
        # Close any earlier plots
        plt.close()

        # Initialize widget parameters
        self.widget_on = True
        self.active_widget = self
        self.prev_widget = self
        self.menu,self.sub_menu,self.info_menu = widgets.HBox(),widgets.HBox(),widgets.HBox()

        # Check for an load dictionary
        if load_file is not None:
            plt.ioff()
            load_data = cp.deepcopy(pkl.load(open(load_file,'rb')))
            # fig = load_data['fig']; ax = fig.axes
            plt.ion()
        else: load_data = None

        # Initialize figure
        self.fig_init(fig,ax)

        # Initialize artist list
        self.artists = dict.fromkeys(self.fig.axes,{})
        # Import the data from the initialized instance
        self.artists[self.ax]['Input Data'],self.input_data = self.setup_data(data)
        self.artists['Primary Artists'] = {self.ax:'Input Data'}
        self.artists['Interactive Axes'] = [self.ax]

        # Initialize data list
        self.data = dict.fromkeys(self.fig.axes,{})

        # Initialize all sub_menus
        self.baseline = baseline(self.fig,menu=self.sub_menu,artists_global=self.artists,data_global=self.data,load_data=load_data)
        self.int_peaks = int_peaks(self.fig,menu=self.sub_menu,artists_global=self.artists,data_global=self.data,load_data=load_data)
        self.fit = fit(self.fig,menu=self.sub_menu,param_menu=self.info_menu,artists_global=self.artists,data_global=self.data,load_data=load_data)
        self.labels = labels(self.fig,menu=self.sub_menu,artists_global=self.artists,data_global=self.data,load_data=load_data)
        self.display_tools = display_tools(self.fig,menu=self.sub_menu,artists_global=self.artists)
        self.export = export(self.fig,main_menu=self,menu=self.sub_menu,param_menu=self.info_menu,exp_path=exp_path)

        # Widget list
        self.widgets = {
        'baseline':self.baseline,
        'fit':self.fit,
        'int_peaks':self.int_peaks,
        'labels':self.labels,
        'display_tools':self.display_tools,
        'export':self.export
        }

        # Setup the main menu and a blank sub_menu
        self.button_list = self.setup_buttons()
        self.place_menu(self.menu,self.button_list)
        self.place_menu(self.sub_menu,[])
        self.place_menu(self.info_menu,[])
        # self.fig_show(self.fig)
        # plt.show()

    # Define all main menu buttons and their functionality
    def setup_buttons(self):

        ##########################################
        ## ON OFF BUTTON
        ##########################################

        # Add an on/off toggle to activate or deactive the widget
        self.on_off_button = widgets.ToggleButton(description='On',value=True,layout = widgets.Layout(width='45px'))

        def on_on_off_button_clicked(b, self = self):

            # If the widget is on, turn it off
            if self.widget_on:
                # Set the description
                self.on_off_button.description = 'Off'
                # Turn off the main widget
                self.deactivate(main=True)
                # Turn off all buttons in the main menu
                for tb in self.button_list:
                    # Turn it off if except for the selected menu
                    if tb != self.on_off_button: tb.value = False
                # Deactivate the current active widget
                if self.active_widget != self: self.active_widget.deactivate()
            # If the widget is off, turn it on
            else:
                self.on_off_button.description = 'On'
                self.activate()

        self.on_off_button.observe(on_on_off_button_clicked)

        ##########################################
        ## BUTTON TEMPLATE
        ##########################################

        def menu_button(self,menuclass,button):
            '''
            Generic button function for all new buttons
            '''
            def func(b,self=self):
                if not self.widget_on:
                    for tb in self.button_list:
                        if tb != self.on_off_button: tb.value = False
                    return
                # If the widget is on, turn it off
                if menuclass.widget_on:
                    # Turn off the widget
                    menuclass.deactivate()
                    # Set the previous widget
                    self.prev_widget = menuclass
                    # Set the active widget to the main menu
                    self.active_widget = self
                # If the widget is off, turn it on
                else:
                    # For each button in the main menu
                    for tb in self.button_list:
                        # Turn it off if except for the selected menu
                        if tb != button and tb != self.on_off_button: tb.value = False
                    # Set the active widget to the selected menu
                    self.active_widget = menuclass
                    # If moving to the export menu
                    if self.active_widget == self.display_tools: self.display_tools.update(self.artists,None)
                    # Update the menu
                    self.prev_widget.update(self.artists,self.data)
                    # Turn off all the menu's buttons
                    for tb in menuclass.toggle_buttons: tb.value = False
                    # Turn on the selected menu
                    menuclass.activate()

                    # Listen for a click
                    self.listen(menuclass)
                self.place_menu(self.info_menu,[])
            return func

        ##########################################
        ## ADD WIDGETS
        ##########################################

        # Baseline toggle
        self.baseline_button = widgets.ToggleButton(description='Baseline',value=False)
        self.baseline_button.observe(menu_button(self,self.baseline,self.baseline_button), names=['value'])

        ##########################################

        # Peaks toggle
        self.int_peaks_button = widgets.ToggleButton(description='Integrate/Peaks',value=False)
        self.int_peaks_button.observe(menu_button(self,self.int_peaks,self.int_peaks_button), names=['value'])

        # ##########################################

        # Integrate toggle
        self.fit_button = widgets.ToggleButton(description='Fit',value=False)
        self.fit_button.observe(menu_button(self,self.fit,self.fit_button), names=['value'])

        # ##########################################

        # Labels toggle
        self.labels_button = widgets.ToggleButton(description='Labels',value=False)
        self.labels_button.observe(menu_button(self,self.labels,self.labels_button), names=['value'])

        # ##########################################
        #
        # Display toggle
        self.display_tools_button = widgets.ToggleButton(description='Display',value=False)
        self.display_tools_button.observe(menu_button(self,self.display_tools,self.display_tools_button), names=['value'])

        # ##########################################
        #
        # Export toggle
        self.export_button = widgets.ToggleButton(description='Export',value=False)
        self.export_button.observe(menu_button(self,self.export,self.export_button), names=['value'])

        ##########################################
        ## BUTTON LIST
        ##########################################

        return [self.on_off_button,self.baseline_button,self.int_peaks_button,self.fit_button,self.labels_button,self.display_tools_button,self.export_button]

    def setup_data(self,data):
        # Setup data
        data_dict = {}
        if isinstance(data,mpl.lines.Line2D):
            data_dict['xpoints'],data_dict['ypoints'] = data.get_data()
            artist = data
        elif isinstance(data,list) or isinstance(data,tuple) or isinstance(data,np.ndarray):
            if isinstance(data[0],mpl.lines.Line2D):
                data_dict['xpoints'],data_dict['ypoints'] = data[0].get_data()
                artist = data[0]
            elif len(data) == 2:
                data_dict['xpoints'],data_dict['ypoints'] = data
                artist = self.ax.plot(data_dict['xpoints'],data_dict['ypoints'])[0]
        return artist,data_dict

    # Initialize all the default colors and color targets
    def color_init(self):
        self.color_target = 'Marker - Primary'
        self.primary_color = 'deepskyblue'
        self.new_color = 'red'
        self.secondary_color = 'green'
        self.fit_color = 'darkviolet'
        self.bscolor = 'blue'
        self.fill_color = 'skyblue'
        self.marker = '*'

    def data_tree(self,*kwargs):
        self.active_widget.update(self.artists,self.data)
        ddict = self.get_data()
        for arg in kwargs:
            if arg in ddict.keys():
                ddict = ddict[arg]
        self.printdict(ddict)

    def artist_tree(self,*kwargs):
        self.active_widget.update(self.artists,self.data)
        adict = self.get_artists()
        for arg in kwargs:
            if arg in adict.keys():
                adict = adict[arg]
        self.printdict(adict)

    def get_data(self,*kwargs):
        # Update the data
        self.active_widget.update(self.artists,self.data)
        # Get the list of axes
        axes = self.get_axes()
        # Create a new data dictionary for data export
        exp_data = {}
        # Fill the new dictionary
        for ind,ax in enumerate(axes):
            if ax in self.artists.keys():
                exp_data[f'Axis {ind+1}'] = self.data[ax]
        for arg in kwargs:
            if arg in exp_data.keys():
                exp_data = exp_data[arg]
        # Return the export dictionary
        return exp_data

    def get_artists(self,*kwargs):
        # Update the artists
        self.active_widget.update(self.artists,self.data)
        # Get the list of axes
        axes = self.get_axes()
        # Create a new data dictionary for data export
        exp_artists = {}
        # Fill the new dictionary
        for ind,ax in enumerate(axes):
            if ax in self.artists.keys():
                exp_artists[f'Axis {ind+1}'] = self.artists[ax]
        for arg in kwargs:
            if arg in exp_artists.keys():
                exp_artists = exp_artists[arg]
        # Return the export dictionary
        return exp_artists

    # def export_dict(self):
    #     self.active_widget.update(self.artists,self.data)
    #     export = {
    #     'artists_global': self.artists,
    #     'data_global': self.data,
    #     }
    #     for key in self.widgets.keys():
    #         if key == 'export' or key == 'display_tools':continue
    #         w_export = {
    #         'data': self.widgets[key].get_data(),
    #         'artists': self.widgets[key].get_artists(),
    #         'style': self.widgets[key].style,
    #         'info': self.widgets[key].info
    #         }
    #         export[key] = w_export
    #
    #     export['fig'] = self.fig
    #
    #     return export

    def export_dict(self):
        self.active_widget.update(self.artists,self.data)
        export = {}

        for key in self.widgets.keys():
            if key == 'export' or key == 'display_tools':continue
            w_export = {
            'data': self.widgets[key].get_data(),
            'style': self.widgets[key].style,
            'info': self.widgets[key].info
            }
            export[key] = w_export

        return export


'''

'''

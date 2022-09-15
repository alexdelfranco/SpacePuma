# Author: Alex DelFranco
# Advisor: Rafa Martin Domenech
# Institution: Center for Astrophysics | Harvard & Smithsonian
# Date: 22 August 2022

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import ipywidgets as widgets

import importlib as imp
from base import widget_base
from baseline import baseline
from integrate import integrate
from peaks import peaks

from scipy import integrate as scpyintegrate
import functools

class main_menu(widget_base):
    def __init__(self,fig=None,ax=None,data=None):
        '''
        Initializes a widget instance
        '''
        # Initialize widget parameters
        self.widget_on = True
        self.active_widget = self
        self.prev_widget = self
        self.menu, self.sub_menu = widgets.HBox(),widgets.HBox()

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
        self.baseline = baseline(self.fig,self.ax,menu=self.sub_menu,artists_global=self.artists,data_global=self.data)
        self.integrate = integrate(self.fig,menu=self.sub_menu,artists_global=self.artists,data_global=self.data)
        self.peaks = peaks(self.fig,menu=self.sub_menu,artists_global=self.artists,data_global=self.data)

        # Setup the main menu and a blank sub_menu
        self.button_list = self.setup_buttons()
        self.place_menu(self.menu,self.button_list)
        self.place_menu(self.sub_menu,[])

    # Define all main menu buttons and their functionality
    def setup_buttons(self):

        ##########################################
        ## ON OFF BUTTON
        ##########################################

        # Add an on/off toggle to activate or deactive the widget
        self.on_off_button = widgets.ToggleButton(description='Widget is On',value=True)

        def on_on_off_button_clicked(b, self = self):

            # If the widget is on, turn it off
            if self.widget_on:
                self.on_off_button.description = 'Widget is Off'
                self.widget_on = False
            # If the widget is off, turn it on
            else:
                self.on_off_button.description = 'Widget is On'
                self.widget_on = True

        self.on_off_button.observe(on_on_off_button_clicked)

        ##########################################
        ## BUTTON TEMPLATE
        ##########################################

        def menu_button(self,menuclass,button):
            '''
            Generic button function for all new buttons
            '''
            def func(b,self=self):
                # If the widget is on, turn it off
                if menuclass.widget_on:
                    # Turn off the widget
                    menuclass.widget_on = False
                    # Set the previous widget
                    self.prev_widget = menuclass
                    # Set the active widget to the main menu
                    self.active_widget = self
                    # Remove the active sub_menu
                    self.place_menu(self.sub_menu,[])
                # If the widget is off, turn it on
                else:
                    # For each button in the main menu
                    for b in self.button_list:
                        # Turn it off if except for the selected menu
                        if b != button and b != self.on_off_button: b.value = False
                    # Turn on the selected menu
                    menuclass.widget_on = True
                    # Set the active widget to the selected menu
                    self.active_widget = menuclass
                    # Update the menu
                    self.prev_widget.update(self.artists,self.data)
                    # Turn off all the menu's buttons
                    for b in menuclass.toggle_buttons: b.value = False
                    # Place the buttons for the menu
                    self.place_menu(self.sub_menu,menuclass.button_list)
                    # Listen for a click
                    self.listen(menuclass)
            return func

        ##########################################
        ## ADD WIDGETS
        ##########################################

        # Baseline toggle
        self.baseline_button = widgets.ToggleButton(description='Baseline',value=False)
        self.baseline_button.observe(menu_button(self,self.baseline,self.baseline_button), names=['value'])

        ##########################################

        # Peaks toggle
        self.peaks_button = widgets.ToggleButton(description='Peaks',value=False)
        self.peaks_button.observe(menu_button(self,self.peaks,self.peaks_button), names=['value'])

        # ##########################################

        # Integrate toggle
        self.integrate_button = widgets.ToggleButton(description='Integrate',value=False)
        self.integrate_button.observe(menu_button(self,self.integrate,self.integrate_button), names=['value'])

        # ##########################################
        #
        # # Figure toggle
        # self.figure_button = widgets.ToggleButton(description='Figure',value=False)
        # self.figure_button.observe(menu_button(self,self.figure,self.figure_button), names=['value'])
        #
        # ##########################################
        #
        # # Labels toggle
        # self.labels_button = widgets.ToggleButton(description='Labels',value=False)
        # self.labels_button.observe(menu_button(self,self.labels,self.labels_button), names=['value'])
        #
        # ##########################################
        #
        # # Export toggle
        # self.export_button = widgets.ToggleButton(description='Export',value=False)
        # self.export_button.observe(menu_button(self,self.export,self.export_button), names=['value'])

        ##########################################
        ## BUTTON LIST
        ##########################################

        return [self.on_off_button,self.baseline_button,self.peaks_button,self.integrate_button]#,
            # ,self.figure_button,
            # self.labels_button,self.export_button]

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

# Author: Alex DelFranco
# Advisor: Rafa Martin Domenech
# Institution: Center for Astrophysics | Harvard & Smithsonian
# Date: 22 September 2022

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as widgets
import functools
import pickle as pkl

import pandas as pd

from .base import widget_base

class export(widget_base):
    def __init__(self,fig,main_menu,menu=None,param_menu=None,exp_path=None):
        '''
        '''
        self.widget_on = False
        self.main_menu = main_menu
        self.fig = fig

        # Initialize defaults
        self.info = self.setup_defaults(exp_path)

        # Initialize all buttons
        self.button_list,self.toggle_buttons,self.param_buttons = self.setup_buttons()
        # Place all the menu buttons
        if menu is None:
            self.menu = widgets.HBox()
            self.place_menu(self.menu,self.button_list)
        else: self.menu = menu
        if param_menu is None:
            self.param_menu = widgets.HBox()
            self.place_menu(self.param_menu,self.param_buttons)
        else: self.param_menu = param_menu

    def setup_defaults(self,exp_path):

        if exp_path is None: exp_path = ''
        info = {
        'exp_path':exp_path,
        'dpi':200,
        'axis': self.fig.axes[0],
        'param_show':False,
        'datadict_path':[]
        }

        return info

    def setup_buttons(self):

        ##########################################
        ## INTERACTIVE PLOTTING BUTTONS
        ##########################################

        # Add a button widget to add new points
        self.export_path_text = widgets.Text(description='Path:',value=self.info['exp_path'],style={'description_width': 'initial'},layout=widgets.Layout(width='200px'))

        def export_path_entered(b, self = self):

            self.info['exp_path'] = self.export_path_text.value

        self.export_path_text.observe(functools.partial(export_path_entered, self=self))

        #####################

        # Add a button widget to add new points
        self.export_all_button = widgets.Button(description='Export Figure')

        def on_export_all_button_clicked(b, self = self):

            exp_dict = self.main_menu.export_dict()
            pkl.dump(exp_dict,open(f"{self.info['exp_path']}.pkl",'wb'))

        self.export_all_button.on_click(functools.partial(on_export_all_button_clicked, self=self))

        #####################

        # Add a button widget to save specfic data
        self.export_data_button = widgets.ToggleButton(description='Export Data')

        def on_export_data_button_clicked(b, self = self):

            if self.info['param_show']:
                self.info['param_show'] = False
                self.place_menu(self.param_menu,[])
            else:
                self.info['param_show'] = True
                self.param_buttons[0].options = ['All'] + list(self.main_menu.get_data().keys())
                self.place_menu(self.param_menu,[self.param_buttons[0]]+[self.save_data_button])

        self.export_data_button.observe(functools.partial(on_export_data_button_clicked, self=self))

        #####################

        # Add a button widget to save the figure
        self.save_button = widgets.Button(description='Save Plot')

        def on_save_button_clicked(b, self = self):

            plt.savefig(f"{self.info['exp_path']}.png",dpi=self.info['dpi'])

        self.save_button.on_click(functools.partial(on_save_button_clicked, self=self))

        #####################

        # Add a button widget to add new points
        self.dpi_text = widgets.IntText(description='DPI:',value=self.info['dpi'],style={'description_width': 'initial'},layout=widgets.Layout(width='100px'))

        def dpi_entered(b, self = self):

            self.info['dpi'] = self.dpi_text.value

        self.dpi_text.observe(functools.partial(dpi_entered, self=self))

        #####################

        self.param_selects = []



        #####################

        # Add a button widget to save the figure
        self.save_data_button = widgets.Button(description='Save Data')

        def on_save_data_button_clicked(b, self = self):

            export_dict = self.main_menu.get_data(*self.info['datadict_path'])
            if type(export_dict) == dict:
                export_df = self.df(export_dict)
            elif type(export_dict) == list or type(export_dict) == np.ndarray:
                export_df = pd.DataFrame()
                export_df[self.info['datadict_path'][-1]] = export_dict
            else: return

            path = self.info['exp_path']

            # If there is no path, do nothing
            if path == '': return
            # If there is no specified extension, default to csv
            if '.' not in path: path += '.csv'

            # Save the data to the desired filetype
            if path.split('.')[-1] == 'csv':
                export_df.to_csv(path)
            elif path.split('.')[-1] == 'dpt':
                self.save_dpt(path,export_df)

        self.save_data_button.on_click(functools.partial(on_save_data_button_clicked, self=self))

        #####################

        return [self.export_path_text,self.export_data_button,self.export_all_button,self.save_button,self.dpi_text],[self.export_data_button],self.param_selects

    def update(self,artists,data): return

    # Define a function used to setup the export buttons
    def data_select(self,param_selects,index):
        # Define a function for the first few buttons
        def func(b,self=self):
            # Make sure the function does not repeatedly get called
            if param_selects[index].value is None: return
            elif param_selects[index].value == 'All':
                self.place_menu(self.param_menu,param_selects[:index+1]+[self.save_data_button])
                return
            # Set the param_select values of the deeper selects to None
            for select in param_selects[index+1:]: select.value = None

            # Clear the data path following the current entry
            self.info['datadict_path'] = self.info['datadict_path'][:index]
            # Add the current entry to the data path
            self.info['datadict_path'].append(param_selects[index].value)

            # Define a subset of the data
            sub_data = self.main_menu.get_data(*self.info['datadict_path'])
            self.sub_data = sub_data
            self.index = index
            self.param_selects = param_selects
            if type(sub_data) == dict and index < len(param_selects)-1:
                # For the next selection, set the option to the keys of the selected dictionary
                param_selects[index+1].options = ['All'] + list(sub_data.keys())
                # Place an updated menu
                self.place_menu(self.param_menu,param_selects[:index+2]+[self.save_data_button])
            else:
                # Place an updated menu
                self.place_menu(self.param_menu,param_selects[:index+1]+[self.save_data_button])
                pass

        return func

    # Override activation method
    def activate(self):
        # Activate the widget
        self.widget_on = True
        self.place_menu(self.menu,self.button_list)
        # Setup parameter selects for saving data
        self.param_selects = []
        for num in range(self.dict_depth(self.main_menu.get_data())):
            button = widgets.Select(options=[],rows=0,style={'description_width': 'initial'},layout = widgets.Layout(width='150px'))
            self.param_selects.append(button)
        # Update the param_buttons as well
        self.param_buttons = self.param_selects
        for index,select in enumerate(self.param_selects):
            select.observe(self.data_select(self.param_selects,index))
        # Return the widget toggle
        return self.widget_on

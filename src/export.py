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

from base import widget_base

class export(widget_base):
    def __init__(self,main_menu,menu=None,exp_path=None):
        '''
        '''
        self.widget_on = False
        self.main_menu = main_menu

        # Initialize defaults
        self.info = self.setup_defaults(exp_path)

        # Initialize all buttons
        self.button_list,self.toggle_buttons = self.setup_buttons()
        # Place all the menu buttons
        if menu is None:
            self.menu = widgets.HBox()
            self.place_menu(self.menu,self.button_list)
        else: self.menu = menu

    def setup_defaults(self,exp_path):

        if exp_path is None: exp_path = ''
        info = {
        'exp_path':exp_path,
        'dpi':200
        }

        return info

    def setup_buttons(self):

        ##########################################
        ## INTERACTIVE PLOTTING BUTTONS
        ##########################################

        # Add a button widget to add new points
        self.export_button = widgets.Button(description='Export')

        def on_export_button_clicked(b, self = self):

            exp_dict = self.main_menu.export_dict()
            pkl.dump(exp_dict,open(f"{self.info['exp_path']}.pkl",'wb'))

        self.export_button.on_click(functools.partial(on_export_button_clicked, self=self))

        #####################

        # Add a button widget to save the figure
        self.save_button = widgets.Button(description='Save Plot')

        def on_save_button_clicked(b, self = self):

            plt.savefig(f"{self.info['exp_path']}.png",dpi=self.info['dpi'])

        self.save_button.on_click(functools.partial(on_save_button_clicked, self=self))

        #####################

        # Add a button widget to add new points
        self.export_path_text = widgets.Text(description='Path:',value=self.info['exp_path'],style={'description_width': 'initial'},layout=widgets.Layout(width='200px'))

        def export_path_entered(b, self = self):

            self.info['exp_path'] = self.export_path_text.value

        self.export_path_text.observe(functools.partial(export_path_entered, self=self))

        #####################

        # Add a button widget to add new points
        self.dpi_text = widgets.IntText(description='DPI:',value=self.info['dpi'],style={'description_width': 'initial'},layout=widgets.Layout(width='100px'))

        def dpi_entered(b, self = self):

            self.info['dpi'] = self.dpi_text.value

        self.dpi_text.observe(functools.partial(dpi_entered, self=self))

        #####################

        return [self.export_button,self.save_button,self.export_path_text,self.dpi_text],[]

    def update(self,artists,data): return

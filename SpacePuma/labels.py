# Author: Alex DelFranco
# Advisor: Rafa Martin Domenech
# Institution: Center for Astrophysics | Harvard & Smithsonian
# Date: 23 September 2022

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import functools

import seaborn as sns

from .base import widget_base

class labels(widget_base):
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
            if 'labels' in load_data: load_data = load_data['labels']

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
        'peak_width':1,
        'selsize':{'Arrows':10,'Labels':40},
        'lab_size':12
        }

        info = {
        # Initialize the add, move, and delete
        'interactive_mode':'off',
        # Initialize an active axis
        'active_ax':None,
        # Initialize a selected boolean
        'selected':False,
        # Set an interactive click distance
        'click_dist':0.04,
        # Save the number of arrows
        'arrow_num':0,
        # Save the number of labels
        'label_num':0,
        # Initialize an arrow base boolean
        'arrow_based':False,
        # Initialize label text
        'label_text':''
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
        self.add_arrow_button = widgets.ToggleButton(description='Add Arrow')

        def on_add_arrow_button_clicked(b, self = self):

            # Turn off the other buttons
            for button in self.toggle_buttons:
                if button != self.add_arrow_button: button.value = False

            # Turn on add_range mode
            if self.info['interactive_mode'] == 'add_arrow':
                self.info['interactive_mode'] = 'off'
            else: self.info['interactive_mode'] = 'add_arrow'

        self.add_arrow_button.observe(functools.partial(on_add_arrow_button_clicked, self=self))

        #####################

        # Add a button widget to add new points
        self.add_label_button = widgets.ToggleButton(description='Add Label')

        def add_label_button_clicked(b, self = self):

            # Turn off the other buttons
            for button in self.toggle_buttons:
                if button != self.add_label_button: button.value = False

            # Turn on integrate mode
            if self.info['interactive_mode'] == 'add_label':
                self.info['interactive_mode'] = 'off'
            else: self.info['interactive_mode'] = 'add_label'

        self.add_label_button.observe(functools.partial(add_label_button_clicked, self=self))

        #####################

        # Add a button widget to add new points
        self.label_text = widgets.Text(description='Label:',style={'description_width':'initial'},layout = widgets.Layout(width='115px'))

        def label_text_clicked(b, self = self):

            if self.label_text.value != 0:
                self.info['label_text'] = self.label_text.value
            else: self.info['label_text'] = ''

        self.label_text.observe(functools.partial(label_text_clicked, self=self))

        #####################

        # Add a button widget to add new points
        self.label_size_inttext = widgets.IntText(description='Size:',value=12,style={'description_width':'initial'},layout = widgets.Layout(width='80px'))

        def label_size_inttext_clicked(b, self = self):

            if self.label_size_inttext.value != 0:
                self.style['lab_size'] = self.label_size_inttext.value
            else: self.style['lab_size'] = 12

        self.label_size_inttext.observe(functools.partial(label_size_inttext_clicked, self=self))

        #####################

        # Add a button widget to add new points
        self.adjust_button = widgets.ToggleButton(description='Adjust')

        def on_adjust_button_clicked(b, self = self):

            # Turn off the other buttons
            for button in self.toggle_buttons:
                if button != self.adjust_button: button.value = False

            # Turn on adjust_range mode
            if self.info['interactive_mode'] == 'adjust':
                self.info['interactive_mode'] = 'off'
            else: self.info['interactive_mode'] = 'adjust'

        self.adjust_button.observe(functools.partial(on_adjust_button_clicked, self=self))

        #####################

        # Add a button widget to add new points
        self.delete_button = widgets.ToggleButton(description='Delete')

        def delete_button_clicked(b, self = self):

            # Turn off the other buttons
            for button in self.toggle_buttons:
                if button != self.delete_button: button.value = False

            # Turn on find peaks mode
            if self.info['interactive_mode'] == 'delete':
                self.info['interactive_mode'] = 'off'
            else: self.info['interactive_mode'] = 'delete'

        self.delete_button.observe(functools.partial(delete_button_clicked, self=self))

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

        return [self.add_arrow_button,self.add_label_button,self.label_text,self.label_size_inttext,self.adjust_button,self.delete_button,self.clear_axis_button],[self.add_arrow_button,self.add_label_button,self.adjust_button,self.delete_button,self.clear_axis_button]

    ##########################################
    ## UPDATE METHODS
    ##########################################

    # Method to create a new artist list
    def new_artists(self,ax,style):
        # Find the center of the axis
        xavg = np.mean(ax.get_xlim())
        yavg = np.mean(ax.get_ylim())

        # Find the arrow and label numbers
        arrow_num = self.info['arrow_num']
        label_num = self.info['label_num']

        artists = {}
        # Add artists if arrows or labels have been added
        if arrow_num != 0: artists[f'Arrow {arrow_num}'] = ax.annotate('',xy=[xavg,yavg],arrowprops=dict(arrowstyle="->")),
        if label_num != 0: artists[f'Label {label_num}'] = ax.annotate('',xy=[xavg,yavg],size=self.style['lab_size'])

        return artists

    def data_init(self):
        return {
        'Arrows':{},
        'Labels':{}
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
            # # Initialize the appropriate artists
            self.update_dictionaries(ax,artist_key='Arrow Base',data_key='Arrows')
            self.update(self.artists_global,self.data_global)
            # Add an arrow base to the artists dictionary if it does not yet exist
            if 'Arrow Base' not in self.artists[ax].keys():
                self.artists[ax]['Arrow Base'] = ax.plot([],[],marker='.',linestyle='',color='red',markersize=12)[0]
                self.artists[ax]['New Arrow/Label'] = ax.plot([],[],markeredgecolor='r',marker='o',markerfacecolor='None')[0]
                self.artists[ax]['Selected Arrow/Label'] = ax.plot([],[],markeredgecolor='g',marker='o',markerfacecolor='None')[0]

        # If a click is not within the axis, do nothing
        if self.info['active_ax'] == None: return
        else: ax = self.info['active_ax']

        # Clear all extra artists
        self.clear(self.artists[ax]['New Arrow/Label'],ax=ax)
        self.clear(self.artists[ax]['Selected Arrow/Label'],ax=ax)

        ##########################################
        ## ADD ARROW
        ##########################################

        # If in add range mode
        if self.info['interactive_mode'] == 'add_arrow':

            if self.info['arrow_based']:

                # Get the arrow number that is being added to
                arrow_num = self.info['arrow_num']
                # Save the data of the arrow head
                arrow_dat = self.data[ax]['Arrows'][f'Arrow {arrow_num}']
                arrow_dat['Head'] = {'xdata':event.xdata,'ydata':event.ydata}
                # Remove the base point artist
                self.clear(self.artists[ax]['Arrow Base'],ax=ax)
                # Flip the arrow boolean
                self.info['arrow_based'] = False

                # Add an arrow annotation
                basex = self.data[ax]['Arrows'][f'Arrow {arrow_num}']['Base']['xdata']
                basey = self.data[ax]['Arrows'][f'Arrow {arrow_num}']['Base']['ydata']

                # Set the head of the arrow
                self.artists[ax][f'Arrow {arrow_num}'] = ax.annotate('',xy=(event.xdata,event.ydata),xytext=(basex,basey),arrowprops=dict(arrowstyle="->"))

            # If this is a new arrow
            else:
                # Increase the number of complete arrows
                self.info['arrow_num'] += 1
                arrow_num = self.info['arrow_num']
                # Flip the arrow boolean
                self.info['arrow_based'] = True
                # Save the location of the arrow base
                self.data[ax]['Arrows'][f'Arrow {arrow_num}'] = {'Base':{'xdata':event.xdata,'ydata':event.ydata}}
                # Plot a point where the base is
                self.artists[ax]['Arrow Base'].set_data(event.xdata,event.ydata)

        ##########################################
        ## ADD LABEL
        ##########################################

        # If in add label mode
        if self.info['interactive_mode'] == 'add_label':

            # If the label is going to be empty, dont bother
            if self.info['label_text'] == '': return
            # Increment the number of labels
            self.info['label_num'] += 1
            # Get the arrow number that is being added to
            label_num = self.info['label_num']
            # Return the label offsets
            xoffset,yoffset = self.get_label_centers(self.info['label_text'],event,ax)
            # Replot the label with the offsets applied
            self.artists[ax][f'Label {label_num}'] = ax.annotate(self.info['label_text'],xy=(event.xdata+xoffset,event.ydata+yoffset/2),size=self.style['lab_size'])
            # Save label location and text data
            self.data[ax]['Labels'][f'Label {label_num}'] = {'xdata':event.xdata,'ydata':event.ydata-yoffset/2,'label_text':self.info['label_text']}

        ##########################################
        ## ADJUST ARROWS/LABELS
        ##########################################

        elif self.info['interactive_mode'] == 'adjust':

            # If an artist is not selected
            if not self.info['selected']:

                # If there was a half added arrow, clear the data
                self.clear(self.artists[ax]['Arrow Base'],ax=ax)
                # Flip the arrow boolean
                self.info['arrow_based'] = False

                # Get all the arrow and label locations
                xlocs,ylocs,labels = self.get_dists(ax)

                # Calculate the distance from the click to each point
                euc = self.bigeuc(event.xdata,event.ydata,xlocs,ylocs,ax)
                # Find the index of the point closest to the click
                self.info['close_index'] = np.nanargmin(euc)

                # If the point is close to the click
                if euc[self.info['close_index']] < self.info['click_dist']:

                    # Plot a circle on top of the selected arrow
                    if 'Arrow' in labels[self.info['close_index']]:
                        self.artists[ax]['Selected Arrow/Label'].set_data(xlocs[self.info['close_index']],ylocs[self.info['close_index']])
                        self.artists[ax]['Selected Arrow/Label'].set(markersize=self.style['selsize']['Arrows'])

                    # Or plot on top of the selected label
                    if 'Label' in labels[self.info['close_index']]:

                        # Calculate the center offsets for the label
                        xoffset,yoffset = self.get_label_centers(self.data[ax]['Labels'][labels[self.info['close_index']]]['label_text'],event,ax)

                        # Plot the selected circle
                        self.artists[ax]['Selected Arrow/Label'].set_data(xlocs[self.info['close_index']],ylocs[self.info['close_index']]+yoffset/4)
                        self.artists[ax]['Selected Arrow/Label'].set(markersize=self.style['selsize']['Labels'])

                    # State that an artist has been selected
                    self.info['selected'] = True

            # If an artist has been selected
            else:
                # Get all the arrow and label locations
                xlocs,ylocs,labels = self.get_dists(ax)

                if 'Arrow' in labels[self.info['close_index']]:
                    # Set a label type
                    lab_type = 'Arrows'
                    # Find the name and base/head value of the arrow
                    arr_name,end = labels[self.info['close_index']].split(':')
                    # Find the data dictionary corresponding to the selected arrow
                    labdata = self.data[ax][lab_type][arr_name][end[1:]]
                    # Set the new location data
                    labdata['xdata'],labdata['ydata'] = event.xdata,event.ydata

                    # Get the data for the arrow
                    if end == ' Head':
                        hxdata,hydata = event.xdata,event.ydata
                        bxdata,bydata = self.data[ax][lab_type][arr_name]['Base']['xdata'],self.data[ax][lab_type][arr_name]['Base']['ydata']
                    elif end == ' Base':
                        hxdata,hydata = self.data[ax][lab_type][arr_name]['Head']['xdata'],self.data[ax][lab_type][arr_name]['Head']['ydata']
                        bxdata,bydata = event.xdata,event.ydata

                    # Delete the previous arrow artist
                    self.artists[ax][arr_name].remove()
                    # Replot the arrow
                    self.artists[ax][arr_name] = ax.annotate('',xy=(hxdata,hydata),xytext=(bxdata,bydata),arrowprops=dict(arrowstyle="->"))

                    # Set the new artist offset to 0
                    yoffset = 0

                else:
                    # Set a label type
                    lab_type = 'Labels'
                    # Find the name of the label
                    lab_name = labels[self.info['close_index']]
                    # Find the data dictionary corresponding to the selected label
                    labdata = self.data[ax][lab_type][lab_name]

                    # Calculate the center offsets for the label
                    xoffset,yoffset = self.get_label_centers(labdata['label_text'],event,ax)

                    # Set the new location data
                    labdata['xdata'],labdata['ydata'] = event.xdata,event.ydata-yoffset/2

                    # Delete the previous label artist
                    self.artists[ax][lab_name].remove()

                    # Replot the label
                    self.artists[ax][lab_name] = ax.annotate(labdata['label_text'],xy=(event.xdata+xoffset,event.ydata+yoffset/2),size=self.style['lab_size'])

                # Plot a red circle around the new arrow
                self.artists[ax]['New Arrow/Label'].set_data(event.xdata,event.ydata-yoffset/4)
                self.artists[ax]['New Arrow/Label'].set(markersize=self.style['selsize'][lab_type])

                # No point is currently selected
                self.info['selected'] = False

        ##########################################
        ## DELETE POINTS
        ##########################################

        # If the delete button has been pressed
        elif self.info['interactive_mode'] == 'delete':

            # Set the selected boolean to false
            self.info['selected'] = False

            # Get all the arrow and label locations
            xlocs,ylocs,labels = self.get_dists(ax)

            # Calculate the distance from the click to each point
            euc = self.bigeuc(event.xdata,event.ydata,xlocs,ylocs,ax)
            # Find the index of the point closest to the click
            self.info['close_index'] = np.nanargmin(euc)

            # If the point is close to the click
            if euc[self.info['close_index']] < self.info['click_dist']:

                # Set a label type
                if 'Arrow' in labels[self.info['close_index']]:
                    lab_type = 'Arrows'
                    lab_name = labels[self.info['close_index']].split(':')[0]
                else:
                    lab_type = 'Labels'
                    lab_name = labels[self.info['close_index']]

                # Remove the label from the local data array
                del self.data[ax][lab_type][lab_name]
                # Remove the label from the global data array
                if lab_name in self.data_global[ax][lab_type].keys():
                    del self.data_global[ax][lab_type][lab_name]

                # Remove the artist from the plot
                self.artists[ax][lab_name].remove()
                # Remove the label from the local artist array
                del self.artists[ax][lab_name]
                # Remove the label from the global artist array
                if lab_name in self.artists_global[ax].keys():
                    del self.artists_global[ax][lab_name]


        ##########################################
        ## CLEAR AXIS
        ##########################################

        # If in clear mode
        elif self.info['interactive_mode'] == 'clear':

            for lab_name in list(self.artists[ax].keys()):
                # Remove the artist from the plot
                if 'Arrow ' in lab_name or 'Label ' in lab_name:
                    self.artists[ax][lab_name].remove()
                    # Remove the label from the local artist array
                    del self.artists[ax][lab_name]
                    # Remove the label from the global artist array
                    if lab_name in self.artists_global[ax].keys():
                        del self.artists_global[ax][lab_name]

            for lab_type in ['Arrows','Labels']:
                # Remove the label from the local data array
                del self.data[ax][lab_type]
                # Remove the label from the global data array
                if lab_type in self.data_global[ax].keys():
                    del self.data_global[ax][lab_type]

            # Set the selected boolean to false
            self.info['selected'] = False
            # Reset the arrow number
            self.info['arrow_num'] = 0
            # Reset the label number
            self.info['label_num'] = 0

        ##########################################
        ## UPDATE ARTISTS
        ##########################################

        # Update artists
        if self.show: plt.show()

    def get_dists(self,ax):
        # Define a new dictionary to hold the distances
        dist_dict = {}
        # For Arrows and Labels in the data dictionary
        for arr_lab in self.data[ax].keys():
            # For each arrow or each label
            for key in self.data[ax][arr_lab]:
                # If it is an arrow
                if 'Arrow ' in key:
                    # Save the arrow data to the distance dict
                    dist_dict[f'{key}: Base'] = [self.data[ax][arr_lab][key]['Base']['xdata'],self.data[ax][arr_lab][key]['Base']['ydata']]
                    dist_dict[f'{key}: Head'] = [self.data[ax][arr_lab][key]['Head']['xdata'],self.data[ax][arr_lab][key]['Head']['ydata']]
                # If it is a label
                if 'Label ' in key:
                    # Save the label data to the distance dict
                    dist_dict[key] = [self.data[ax][arr_lab][key]['xdata'],self.data[ax][arr_lab][key]['ydata']]
        # Save the labels for each distance
        labs = list(dist_dict.keys())
        # Zip the x,y tuples into x and y arrays
        x,y = zip(*dist_dict.values())
        # Return the x, y, and label array
        return x,y,labs

    def get_label_offsets(self,label_artist,ax):
        # Get the bounding box bounds for the label artist
        bb = np.array(label_artist.get_tightbbox(None).bounds)
        # Caclulate the figure pixel coords of the lower left and upper right corners
        low_left = bb[:2]; up_right = bb[:2] + bb[2:]
        # Define a figure pixel inversion transformation
        inv = ax.transData.inverted()
        # Average the corners to find the x and y offsets of the box in data coordinates
        xoffset,yoffset = (inv.transform(low_left) - inv.transform(up_right))/2
        # Return the label offsets
        return xoffset,yoffset

    def get_label_centers(self,text,event,ax):
        # Set a dummy label
        dummy_label = ax.annotate(text,xy=(event.xdata,event.ydata),size=self.style['lab_size'],color='None')
        # Calculate offsets from the lower left corner to the label center
        xoffset,yoffset = self.get_label_offsets(dummy_label,ax=ax)
        # Remove the dummy artist from the plot
        dummy_label.remove()
        # Return the label offsets
        return xoffset,yoffset

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
            if 'Arrows' in load_data['data'][axis]:

                # Shorten the path
                arr_dat = load_data['data'][axis]['Arrows']

                # Loop through all the arrows
                arrow_num = 1
                while True:
                    if f'Arrow {arrow_num}' not in arr_dat: break
                    else:
                        adata = arr_dat[f'Arrow {arrow_num}']

                        self.info['interactive_mode'] = 'add_arrow'

                        # Call the __call__ method to add the tail point
                        self(sim_event(adata['Base']['xdata'],adata['Base']['ydata'],ax))

                        # Call the __call__ method to add the head point
                        self(sim_event(adata['Head']['xdata'],adata['Head']['ydata'],ax))

                        arrow_num += 1

            # Store the points which define the baseline
            if 'Labels' in load_data['data'][axis]:

                # Shorten the path
                lab_dat = load_data['data'][axis]['Labels']

                # Loop through all the arrows
                lab_num = 1
                while True:
                    if f'Label {lab_num}' not in lab_dat: break
                    else:
                        ldata = lab_dat[f'Label {lab_num}']

                        self.info['interactive_mode'] = 'add_label'
                        self.info['label_text'] = ldata['label_text']

                        # Call the __call__ method to add the tail point
                        self(sim_event(ldata['xdata'],ldata['ydata'],ax))

                        lab_num += 1

        self.widget_on = False
        self.show = True
        self.info['interactive_mode'] = 'off'

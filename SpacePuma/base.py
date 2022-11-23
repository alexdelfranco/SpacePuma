# Author: Alex DelFranco
# Advisor: Rafa Martin Domenech
# Institution: Center for Astrophysics | Harvard & Smithsonian
# Date: 22 August 2022

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

import pandas as pd

class widget_base:
    '''
    This class serves as the base class for all of the astrochem-tools
    interactive classes. The methods outlined in this class will be inhereted
    and modified to enable different interactive elements in the other widgets.
    '''
    def __init__(self,fig=None,ax=None): return

    def __call__(self,event): return

    ##########################################
    ## SETUP METHODS
    ##########################################

    def setup_buttons(self): return

    # Update the buttons onscreen
    def place_menu(self,menu,button_list):
        # Place the buttons and menus into a box and display it
        menu.children = button_list
        # Set the output
        output = widgets.Output()
        # Display the box
        display(menu, output)
        self.fig.canvas.draw_idle()

    # Initialize Artists
    def define_artists(self): return

    ##########################################
    ## MATH METHODS
    ##########################################

    # Calculate the eucldian distance of a point to a series of points
    def bigeuc(self,x0,y0,xarr,yarr,ax):
        # Convert array variables into numpy arrays for arithmetic
        xarr = np.array(xarr); yarr = np.array(yarr)
        # Find the boundaries of the axis
        xmin,xmax = ax.get_xlim(); ymin,ymax = ax.get_ylim()
        # Subtract off the minima to set the lower bound at 0
        x0 -= xmin; xarr -= xmin; xmax -= xmin
        y0 -= ymin;
        yarr -= ymin;
        ymax -= ymin
        # Normalize all values to range from 0 to 1
        x0 /= xmax; xarr /= xmax; y0 /= ymax; yarr /= ymax
        # Return the euclidian distances of the normalized points
        return [self.euclid(x0,y0,xarr[ii],yarr[ii]) for ii in range(len(xarr))]

    # Calculate the euclidian distance between two points
    def euclid(self,x0, y0, x1, y1):

        return np.sqrt((x1 - x0)**2 + (y1-y0)**2)

    # Calculate the horizontal distance between a point and an array of points
    def hdist(self,x0,xarr,ax):
        return self.bigeuc(x0,0,xarr,np.zeros_like(xarr),ax)

    # Find a y value for an x value along a line defined by two points
    def lincalc(self,x1,y1,x2,y2,x3):
        m = (y2-y1)/(x2-x1)
        y3 = m * (x3-x1) + y1
        return y3

    ##########################################
    ## SUPPORT METHODS
    ##########################################

    # Calculate ordered arrays from the xdat and ydat arrays
    def order(self,xdata,ydata):
        # Sort the data into tuples so that the line connects linearly
        xytuples = sorted([i for i in zip(xdata,ydata)])
        # Create new arrays to store the separated x and y values
        xord,yord = [],[]
        # Separate the ordered x and y values
        for x,y in xytuples: xord.append(x); yord.append(y)

        return(xord,yord)

    # Listen for a button press event
    def listen(self,widget):
        # Call a listener on the point
        self.cid = self.fig.canvas.mpl_connect('button_press_event',widget)

    # Clear a plot
    def clear(self,line,vlines=False,ax=None):
        # If there was a newpoint added
        if line != None and not vlines:
            # Remove all points
            line.set_data([],[])
        if vlines:
            # Remove all vlines
            self.set_segments(line,[],ax)

    ##########################################
    ## FIGURE METHODS
    ##########################################

    def fig_init(self,fig,ax):
        if (fig is None) or (ax is None):
            # Close any past figures
            plt.close()
            # Define the default figure size for the plot
            self.figsize = [10,3.5]
            # Define new axes
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
            self.fig.tight_layout()
        else:
            # Use predefined fig and ax
            self.fig,self.ax = fig,ax
            # Determine the user-defined figure size
            self.figsize = self.fig.get_size_inches()

    def resize_figure(self,nplots,fig):
        '''
        Changes the number of vertical axes in a matplotlib figure
        with dimensions 10 by 3.5 * nplots inches and adjusts the
        figure size and axes placement automatically

        Parameters
        ----------
        nplots : int
            An integer denoting the final number of axes the figure
            will have after resizing
        fig : matplotlib.figure
            The matplotlib figure object being resized
        axes : matplotlib.axes or [matplotlib.axes, ...]
            The matplotlib axes object or list of matplotlib axes objest
            being resized
        '''
        # Create axes variable
        axes = fig.axes
        # Create a temporary figure
        temp_fig,temp_axes = plt.subplots(nplots,1,figsize=(10,3.5*nplots))
        temp_fig.tight_layout()
        # Save bounds from that temporary figure
        if nplots==1: bounds = [temp_axes.get_position().bounds]
        else: bounds = [temp_ax.get_position().bounds for temp_ax in temp_axes]
        # If there is only one axis
        if str(type(axes)) == "<class 'matplotlib.axes._subplots.AxesSubplot'>":
            # Resize it
            axes.set_position(bounds[0])
            # Resize the rest of the axes
            for bound in bounds[1:]: fig.add_axes(bound)
        # If there are more than one axis
        else:
            # If the number of axes are being reduced
            if len(axes) > nplots:
                # Delete all the extra axes
                for ax in axes[nplots:]: fig.delaxes(ax)
                # Redefine the input to a truncated version
                axes = fig.axes
            # Resize the rest of the axes
            for ind,ax in enumerate(axes): ax.set_position(bounds[ind])
            for bound in bounds[len(axes):]: fig.add_axes(bound)
        # Resize the figure
        fig.set_size_inches(10,3.5*nplots)
        # Close the temporary figure
        plt.close(temp_fig)
        # Return the new axes
        return([fig.get_axes(),bounds,axes,nplots])

    def set_segments(self,vline,arr,ax):
        ymin,ymax = ax.get_ylim()
        newsegs = [[[xval,ymin],[xval,ymax]] for xval in arr]
        vline.set_segments(newsegs)

    def update(self,artists,data):
        '''
        Called every time a new menu is selected
        It should pull the artists and data into the primary dicts
        The artist_ and data_ push methods should be called as each are being
        updated, so the primaries should always be kept up to date
        Just remember to delete any artists/data from both local and primary dicts
        '''
        # Update the global artist and data dictionaries
        self.push_artists(artists,self.artists)
        self.push_data(data,self.data)

    def push_artists(self,artists_global,artists):
        for ax in artists:
            if ax not in artists_global.keys(): artists_global[ax] = {}
            for artist in artists[ax]:
                if isinstance(ax,str): artists_global[ax] = artists[ax]
                else: artists_global[ax][artist] = artists[ax][artist]

    def pull_artists(self,artists=None):
        if artists is None: return {'Primary Artists':{},'Interactive Axes':{}}
        else: return artists

    def push_data(self,data_global,data):
        for ax in data:
            if ax not in data_global.keys(): data_global[ax] = {}
            for dat in data[ax]:
                data_global[ax][dat] = data[ax][dat]

    def pull_data(self,data=None):
        if data is None: return {'Primary Artists':{},'Interactive Axes':{}}
        else: return data

    def update_dictionaries(self,ax,artist_key,data_key):
        # LOCAL ARTISTS
        if ax not in self.artists.keys():
            # Add the axis to the artist dict
            self.artists[ax] = self.new_artists(ax,self.style)
        elif artist_key not in self.artists[ax].keys():
            # Else, merge the dictionary with a new one
            self.artists[ax] = self.artists[ax] | self.new_artists(ax,self.style)
        # GLOBAL ARTISTS
        if ax not in self.artists_global.keys():
            # Add the axis to the artist dict
            self.artists_global[ax] = self.new_artists(ax,self.style)
        elif artist_key not in self.artists_global[ax].keys():
            # Else, merge the dictionary with a new one
            self.artists_global[ax] = self.artists_global[ax] | self.new_artists(ax,self.style)
        # LOCAL DATA
        if ax not in self.data.keys():
            # Add the axis to the data dict
            self.data[ax] = self.data_init()
            self.data_global[ax] = self.data_init()
        elif data_key not in self.data[ax].keys():
            # Else, merge the dictionary with a new one
            self.data[ax] = self.data[ax] | self.data_init()
        # GLOBAL DATA
        if ax not in self.data_global.keys():
            # Add the axis to the data dict
            self.data_global[ax] = self.data_init()
        elif data_key not in self.data_global[ax].keys():
            # Else, merge the dictionary with a new one
            self.data_global[ax] = self.data_global[ax] | self.data_init()

    def printdict(self,dictionary,indent=''):
        if indent=='': print('Dictionary')
        # If the input is a dictionary
        if isinstance(dictionary,dict):
            # Save the list of keys
            keys = list(dictionary.keys())
            # Set the indent strings
            tab = ' |  '
            turn = ' |--'
            # For each key of the dictionary
            for key in keys:
                # # Check if it is the final key
                if key == keys[-1]:
                    # Set special indent strings
                    tab = '    '
                    turn = " '--"
                # Print the dictionary key
                print(f'{indent}{turn}{key}')
                # Check to see if it has any subdictionaries
                self.printdict(dictionary[key],indent+tab)
        # Once all sub-dictionaries have been printed, return
        else: return

    def fig_show(self,fig):
        # Create a new figure
        figure = plt.figure()
        # Create a new manager
        manager = figure.canvas.manager
        # Overwrite the figure in the manager
        manager.canvas.figure = fig
        # Set the figure canvas to the manager's canvas
        fig.set_canvas(manager.canvas)

# ADMIN METHODS

    def activate(self):
        # Activate the widget
        self.widget_on = True
        self.place_menu(self.menu,self.button_list)
        # Return the widget toggle
        return self.widget_on

    def deactivate(self,main=False):
        # Deactivate the widget
        self.widget_on = False
        # If not the main menu
        if not main:
            # Clear the widget menu
            self.place_menu(self.menu,[])
        # Return the widget toggle
        return self.widget_on

# FUNCTIONAL METHODS

    def dict_depth(self,dct):
        depths = []
        for key in dct:
            if type(dct[key]) == dict:
                depths.append(self.dict_depth(dct[key]))
        if len(depths) == 0: depths.append(0)
        return max(depths)+1

# PRINT AND RETURN METHODS

    def data_tree(self,*kwargs):
        ddict = self.get_data()
        for arg in kwargs:
            if arg in ddict.keys():
                ddict = ddict[arg]
        self.printdict(ddict)

    def artist_tree(self,*kwargs):
        adict = self.get_artists()
        for arg in kwargs:
            if arg in adict.keys():
                adict = adict[arg]
        self.printdict(adict)

    def get_data(self,*kwargs):
        # Get the list of axes
        axes = self.get_axes()
        # Create a new data dictionary for data export
        exp_data = {}
        # Fill the new dictionary
        for ind,ax in enumerate(axes):
            if ax in self.data:
                exp_data[f'Axis {ind+1}'] = self.data[ax]
        for arg in kwargs:
            if arg in exp_data.keys():
                exp_data = exp_data[arg]
        # Return the export dictionary
        return exp_data

    def get_artists(self,*kwargs):
        # Get the list of axes
        axes = self.get_axes()
        # Create a new data dictionary for data export
        exp_artists = {}
        # Fill the new dictionary
        for ind,ax in enumerate(axes):
            exp_artists[f'Axis {ind+1}'] = self.artists[ax]
        for arg in kwargs:
            if arg in exp_artists.keys():
                exp_artists = exp_artists[arg]
        # Return the export dictionary
        return exp_artists

    def get_fig(self):
        return self.fig

    def get_axes(self):
        return self.fig.axes

    def export_dict(self):
        return {}

    def df(self,datadict,dataframe=pd.DataFrame(),path='',collist=[]):
        '''
        Create a dataframe from a Dictionary
        Recursive method
        '''
        for key in datadict:
            data = datadict[key]
            if type(data) == dict:
                # If the data is a dictionary, iterate recusively
                dataframe = self.df(datadict[key],dataframe,f'{path} - {key}',collist=collist)
            elif type(data) == list or type(data) == type(np.array([])):
                # Create a new dataframe with the list
                newdf = pd.DataFrame(data,columns=[f'{path} - {key}'[3:]])
                cols = list(dataframe.columns)
                # Merge the shorter dataframe to the larger one
                if len(dataframe) > len(newdf): dataframe = dataframe.join(newdf)
                else: dataframe = newdf.join(dataframe)
                # Update the list of columns
                cols.append(list(newdf.columns)[0])
                # Reorder the dataframe
                dataframe = dataframe[cols]
        # Return the final datafame
        return dataframe

    ##########################################
    ## FILETYPE EXPORTS
    ##########################################

    def save_dpt(self,filename,dataframe):
        # Check to make sure the data is of the right type
        if type(dataframe) == dict:
            dataframe = self.df(dataframe)
            self.save_dpt(filename,dataframe)
        elif type(dataframe) == list or type(dataframe) == np.ndarray:
            dataframe = pd.DataFrame()
            dataframe['data'] = dataframe
            self.save_dpt(filename,dataframe)
        elif type(dataframe) == type(pd.DataFrame()):
            with open(filename, "w") as newFile:
                for index in range(len(dataframe)):
                    dataline = ''
                    for point in dataframe.iloc[index]:
                        dataline += f'{point} '
                    newFile.write(f'{dataline[:-1]}\n')
        else: print('Please enter valid data')

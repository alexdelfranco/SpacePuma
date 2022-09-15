# Author: Alex DelFranco
# Advisor: Rafa Martin Domenech
# Institution: Center for Astrophysics | Harvard & Smithsonian
# Date: 25 July 2022

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import functools
from scipy.interpolate import interp1d
from scipy import integrate as scpyintegrate
from scipy.signal import find_peaks,medfilt
from scipy.optimize import curve_fit
import seaborn as sns
import pickle
import copy as cp

class interactive:
    '''
    Parent Interactive Class
    '''
    # Initialize the class
    def __init__(self,fig=None,ax=None,x=[],y=[],menu_count=2):

        # Set the widget to active
        self.widget_on = True
        self.button_group = 0
        self.menu = widgets.HBox()
        self.menu_count = menu_count

        # Initialize a color scheme
        self.color_init()

        # No point is currently selected
        self.selected = False
        # Set the distance a click needs to be from a point to select it
        self.click_dist = 0.02

        # Start on add mode, set all other modes to false
        self.add_mode = False
        self.move_mode = False
        self.delete_mode = False
        self.axscale = 0

        # Initialize the newpoint and selpoint vars
        self.newpoint = None
        self.selpoint = None

        # Initialize the two input arrays
        self.xdat = x
        self.ydat = y

        # Initialize the figure
        self.fig_init(fig,ax)

        # Plot the initial points
        scatter, = self.ax.plot(self.xdat,self.ydat,marker='*',linestyle='',color=self.primary_color,markersize=15)

        # Save the original size of the figure
        self.xbound_min,self.xbound_max = self.ax.get_xlim()
        self.ax.set_yscale('log')

        # Begin listening for clicks
        self.listen(scatter)

        # Setup all buttons
        self.setup_buttons()

        # Setup any custom BUTTONS
        self.setup_custom_buttons()

        # Place all buttons
        self.place_buttons()

    # Initialize the figure
    def fig_init(self,fig,ax):

        if fig == None or ax == None:
            # Close any past figures
            plt.close()
            # Define new axes
            self.fig, self.ax = plt.subplots()
        else:
            # Use predefined fig and ax
            self.fig,self.ax = fig,ax

    # Define all buttons and their functionality
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

        # Add a button widget to switch between menus
        self.menu_button = widgets.Button(description='Change Menu')

        def on_menu_button_clicked(b, self = self):

            # Change between button menus
            self.button_group = (self.button_group + 1) % self.menu_count
            # Draw the new menu
            self.place_buttons()

        self.menu_button.on_click(functools.partial(on_menu_button_clicked, self=self))

        ##########################################
        ## INTERACTIVE PLOTTING BUTTONS
        ##########################################

        # Add a button widget to add new points
        self.add_button = widgets.ToggleButton(description='Add Points')

        def on_add_button_clicked(b, self = self):

            self.move_mode = self.move_button.value = False
            self.delete_mode = self.delete_button.value = False

            if self.add_mode: self.add_mode = False
            else: self.add_mode = True

        self.add_button.observe(functools.partial(on_add_button_clicked, self=self))

        #####################

        # Add a button widget to move points
        self.move_button = widgets.ToggleButton(description='Move Points')

        def on_move_button_clicked(b, self = self):

            self.add_mode = self.add_button.value = False
            self.delete_mode = self.delete_button.value = False

            if self.move_mode: self.move_mode = False
            else: self.move_mode = True

        self.move_button.observe(functools.partial(on_move_button_clicked, self=self))

        #####################

        # Add a button widget to delete points
        self.delete_button = widgets.ToggleButton(description='Delete Points')

        def on_delete_button_clicked(b, self = self):

            self.add_mode = self.add_button.value = False
            self.move_mode = self.move_button.value = False

            if self.delete_mode: self.delete_mode = False
            else: self.delete_mode = True

        self.delete_button.observe(functools.partial(on_delete_button_clicked, self=self))

        #####################

        # Add a button widget to clear all points
        self.clear_all_button = widgets.Button(description='Clear All')

        def on_clear_all_button_clicked(b, self = self):

            # Clear all other points
            self.clear(self.newpoint)
            self.clear(self.selpoint)
            self.selected = False

            # Clear all points
            self.xdat = np.array([])
            self.ydat = np.array([])

            # Replot all the points, excluding the selected one
            self.points.set_data([self.xdat,self.ydat])

        self.clear_all_button.on_click(functools.partial(on_clear_all_button_clicked, self=self))

        #####################

        # Add a button widget to clear all points
        self.hide_marker_button = widgets.ToggleButton(description='Markers On',value=True)

        def on_hide_marker_button_clicked(b, self = self):

            if self.marker == '*':
                self.marker = ''
            else:
                self.marker = '*'

            self.points.set_marker(self.marker)
            try: self.newpoint.set_marker(self.marker)
            except: pass
            try: self.selpoint.set_marker(self.marker)
            except: pass

        self.hide_marker_button.observe(on_hide_marker_button_clicked)

        ##########################################
        ## AXIS SCALING
        ##########################################

        # Add a button widget to change the scale of the axis
        self.scale_button = widgets.Button(description='Set Linear Scale')

        def on_scale_button_clicked(b, self = self):

            self.axscale = (self.axscale + 1) % 2

            if self.axscale == 1:
                self.ax.set_yscale('linear')
                self.scale_button.description = 'Set Log Scale'
                # self.bsax.set_yscale('log')
            elif self.axscale == 0:
                self.ax.set_yscale('log')
                self.scale_button.description = 'Set Linear Scale'

        self.scale_button.on_click(functools.partial(on_scale_button_clicked, self=self))

        ##########################################
        ## COLORS
        ##########################################

        # Add a selection menu to choose what target you want to change the color of
        self.color_target_select = widgets.Select(description='Color Target',options=['Marker - Primary','Marker - Secondary','Marker - New'],rows=0)

        def on_color_target_clicked(b, self = self):

            # Set the color target to the menu selection
            self.color_target = self.color_target_select.value

        self.color_target_select.observe(on_color_target_clicked)

        ##########################################

        # Add a color picker to allow custom colors
        self.color_picker = widgets.ColorPicker(description='Edit Color',concise=True,value=self.primary_color)

        def on_color_picker_clicked(b, self = self):

            # Save the user-selected color
            self.color = self.color_picker.value

            # If primary markers are selected
            if self.color_target == 'Marker - Primary':

                # Set the primary marker color and update the figure
                self.primary_color = self.color
                self.points.set_color(self.color)

            # If secondary markers are selected
            elif self.color_target == 'Marker - Secondary':

                # Set the secondary marker color
                self.secondary_color = self.color

            # If new markers are selected
            elif self.color_target == 'Marker - New':

                # Set the new marker color
                self.new_color = self.color

        self.color_picker.observe(on_color_picker_clicked)

    # Define any custom buttons (meant to be inhereted)
    def setup_custom_buttons(self):
        return

    # Update the buttons onscreen
    def place_buttons(self):
        # Place the buttons and menus into a box and display it
        if self.button_group == 0:
            self.menu.children = [self.on_off_button,self.menu_button,self.add_button,self.move_button,self.delete_button,self.clear_all_button,self.hide_marker_button]
        elif self.button_group == 1:
            self.menu.children = [self.on_off_button,self.menu_button,self.scale_button,self.color_target_select,self.color_picker]
        # Set the output
        output = widgets.Output()
        # Display the box
        display(self.menu, output)
        self.fig.canvas.draw_idle()

    # When the class is called as a function
    def __call__(self, event):

        if not self.widget_on: return
        # If a click is not within the axis, do nothing
        if event.inaxes!=self.points.axes: return

        # If the add button has been pressed
        if self.add_mode:

            # Clear all other points
            self.clear(self.newpoint)
            self.clear(self.selpoint)
            self.selected = False

            # Add the point to the array of points
            self.xdat = np.append(self.xdat,event.xdata)
            self.ydat = np.append(self.ydat,event.ydata)

            # Replot all the points, including the extra one
            self.points.set_data([self.xdat,self.ydat])

        # If the move button has been pressed
        elif self.move_mode:

            # If a point is not selected
            if not self.selected:
                # If there was a newpoint added, clear it
                self.clear(self.newpoint)
                # Calculate the distance from the click to each point
                self.euc = self.bigeuc(event.xdata,event.ydata,self.xdat,self.ydat)
                # Find the index of the point closest to the click
                self.close = np.nanargmin(self.euc)

                # If the point is close to the click
                if self.euc[self.close] < self.click_dist:

                    # Replot the selected point in a different color
                    self.selpoint, = self.ax.plot(self.xdat[self.close],self.ydat[self.close],marker='*',linestyle='',color=self.secondary_color,markersize=15)
                    # State that a point has been selected
                    self.selected = True

            # If a point has already been selected
            else:

                # Remove the point from the data arrays
                self.xdat = np.delete(self.xdat,self.close)
                self.ydat = np.delete(self.ydat,self.close)

                # Remove the temporary plotted point
                self.clear(self.selpoint)

                # Plot the newly placed point
                self.newpoint, = self.ax.plot(event.xdata, event.ydata, color=self.new_color,marker='*', markersize=15)

                # Add the point to the array of points
                self.xdat = np.append(self.xdat,event.xdata)
                self.ydat = np.append(self.ydat,event.ydata)

                # Replot all the points, excluding the selected one
                self.points.set_data([self.xdat,self.ydat])

                # No point is currently selected
                self.selected = False

        # If the delete button has been pressed
        elif self.delete_mode:

            # Clear all other points
            self.clear(self.newpoint)
            self.clear(self.selpoint)
            self.selected = False

            # Calculate the distance from the click to each point
            self.euc = self.bigeuc(event.xdata,event.ydata,self.xdat,self.ydat)
            # Find the index of the point closest to the click
            self.close = np.nanargmin(self.euc)

            # If the point is close to the click
            if self.euc[self.close] < self.click_dist:
                # Remove the point from the data arrays
                self.xdat = np.delete(self.xdat,self.close)
                self.ydat = np.delete(self.ydat,self.close)

                # Replot all the points, excluding the selected one
                self.points.set_data([self.xdat,self.ydat])

            # If there is only one point left, clear the fit
            if len(self.xdat) < 2: self.clearfit()

        # If there is a fit, update it
        self.fit()

    ##########################################
    ## FITTING METHODS
    ##########################################

    def fit(self): return

    def clearfit(self): return

    ##########################################
    ## MATH METHODS
    ##########################################

    # Calculate the eucldian distance of a point to a series of points
    def bigeuc(self,x0,y0,xarr,yarr):
        # Convert array variables into numpy arrays for arithmetic
        xarr = np.array(xarr); yarr = np.array(yarr)
        # Find the boundaries of the axis
        xmin,xmax = self.ax.get_xlim(); ymin,ymax = self.ax.get_ylim()
        # Subtract off the minima to set the lower bound at 0
        x0 -= xmin; xarr -= xmin; xmax -= xmin
        y0 -= ymin; yarr -= ymin; ymax -= ymin
        # Normalize all values to range from 0 to 1
        x0 /= xmax; xarr /= xmax; y0 /= ymax; yarr /= ymax
        # Return the euclidian distances of the normalized points
        return [self.euclid(x0,y0,xarr[ii],yarr[ii]) for ii in range(len(xarr))]

    # Calculate the euclidian distance between two points
    def euclid(self,x0, y0, x1, y1):

        return np.sqrt((x1 - x0)**2 + (y1-y0)**2)

    # Find a y value for an x value along a line defined by two points
    def lincalc(self,x1,y1,x2,y2,x3):
        m = (y2-y1)/(x2-x1)
        y3 = m * (x3-x1) + y1
        return y3

    ##########################################
    ## SUPPORT METHODS
    ##########################################

    # Calculate ordered arrays from the xdat and ydat arrays
    def order(self):
        # Sort the data into tuples so that the line connects linearly
        xytuples = sorted([i for i in zip(self.xdat,self.ydat)])
        # Create new arrays to store the separated x and y values
        xord,yord = [],[]
        # Separate the ordered x and y values
        for x,y in xytuples: xord.append(x); yord.append(y)

        return(xord,yord)

    # Listen for a button press event
    def listen(self, points):

        # Save the points
        self.points = points
        # Call a listener on the point
        self.cid = points.figure.canvas.mpl_connect('button_press_event', self)

    # Clear a plot
    def clear(self,point):
        # If there was a newpoint added
        if point != None:
            # Remove the last new point
            point.set_data([],[])

    ##########################################
    ## RETURN METHODS
    ##########################################

    # Return the plotted points
    def get_points(self):
        return self.xdat,self.ydat

    ##########################################
    ## COLOR
    ##########################################

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

class baseline(interactive):
    '''
    Baselining class which inherets methods from the interactive class
    '''
    def __init__(self,fig=None,ax=None,baseline=None,x=[],y=[],output_path='',load_file=None):

        # Initialize the fit override order
        self.fit_order_override = 0
        # Set a default fit type
        self.fit_type = 'Linear Segment'
        # Set an output path
        self.output_path = output_path
        self.output_filename = 'plots'
        # Check if to include a baseline
        self.baseline = baseline
        self.do_baseline = self.baseline != None

        # If loading in from a file, set the input points accordingly
        if load_file != None:
            self.loaddat = pickle.load(open(load_file,'rb'))
            x,y = self.loaddat['points']['data']

        # Initialize the interactive widget (inhereting all of its methods)
        super().__init__(fig=fig,ax=ax,x=x,y=y,menu_count=4)

        # Initialize the fit
        self.plotfit, = self.ax.plot([],[],color=self.fit_color)

        # Match the axis bounds on the baseline plot
        if self.do_baseline:
            self.bsax.set_xlim(self.xbound_min,self.xbound_max)
            self.bsax.autoscale(axis='y')

        # Plot the fit
        self.fit()

        # Setup the integration widget
        self.int.hide_buttons = False
        self.int.place_buttons()

        # If we are loading in from a file, set the integration points
        if load_file != None:
            self.int.xdat,self.int.ydat = self.loaddat['intpoints']['data']

        # Initialize baseline fig
        self.baseline_update()

    # Initialize the figure
    def fig_init(self,fig,ax):

        if fig == None:
            # Close any past figures
            plt.close()
            # Define new axes
            self.fig, self.ax = plt.subplots()
        elif not self.do_baseline:
            # Use predefined fig and ax
            self.fig,self.ax = fig,ax
        else:
            # THROW ERROR HERE IF AX ONLY HAS ONE ELEMENT
            # Use predefined fig and axes
            self.fig,self.ax,self.bsax = fig,ax[0],ax[1]
            # Save the baseline
            self.bsx,self.bsy = self.baseline
            self.bsx,self.bsy = np.array(self.bsx),np.array(self.bsy)
            # Initialize the baseline
            self.bsplot, = self.bsax.plot([],[],color=self.bscolor)
            # Add integration interaction for baseline plot
            self.int = integrate(self.fig,self.bsax,intdata=([],[]),hide_buttons=True)

    # Setup all custom buttons
    def setup_custom_buttons(self):

        #####################

        # Add a button widget to clear all points
        self.clear_all_button = widgets.Button(description='Clear All')

        def on_clear_all_button_clicked(b, self = self):

            # Clear all other points
            self.clear(self.newpoint)
            self.clear(self.selpoint)
            if self.do_baseline: self.clear(self.bsplot)
            self.selected = False

            # Clear all points
            self.xdat = np.array([])
            self.ydat = np.array([])

            # Replot all the points, excluding the selected one
            self.points.set_data([self.xdat,self.ydat])

            # Clear the fit
            self.plotfit.set_data([],[])

        self.clear_all_button.on_click(functools.partial(on_clear_all_button_clicked, self=self))

        #####################

        # Add a color picker to allow custom colors
        self.color_picker = widgets.ColorPicker(description='Edit Color',concise=True,value=self.primary_color)

        def on_color_picker_clicked(b, self = self):

            # Save the user-selected color
            self.color = self.color_picker.value

            # If primary markers are selected
            if self.color_target == 'Marker - Primary':

                # Set the primary marker color and update the figure
                self.primary_color = self.color
                self.points.set_color(self.color)

            # If secondary markers are selected
            elif self.color_target == 'Marker - Secondary':

                # Set the secondary marker color
                self.secondary_color = self.color

            # If new markers are selected
            elif self.color_target == 'Marker - New':

                # Set the new marker color
                self.new_color = self.color

            # If fit is selected
            elif self.color_target == 'Fit':

                # Set the fit color and update the figure
                self.fit_color = self.color
                self.plotfit.set_color(self.color)

        self.color_picker.observe(on_color_picker_clicked)

        ##########################################
        ## FITTING
        ##########################################

        # Add a dropdown box to select the fit type
        self.fit_type_select = widgets.Select(description='Fit Type',options=['Linear Segment','Polynomial'],rows=0)

        def on_fit_type_selected(b, self = self):

            # Change fit order value
            self.fit_type = self.fit_type_select.value
            # Redraw the buttons
            self.place_buttons()
            # Recalculate the fit
            self.fit()

        self.fit_type_select.observe(functools.partial(on_fit_type_selected, self=self))

        ##########################################

        # Add a text box to set the fit order
        self.fit_order_inttext = widgets.IntText(description='Fit Order',value=0)

        def on_fit_order_inttext_entered(b, self = self):

            # Change fit order value
            self.fit_order_override = self.fit_order_inttext.value
            # Recalculate the fit
            self.fit()

        self.fit_order_inttext.observe(functools.partial(on_fit_order_inttext_entered, self=self))

        ##########################################

        self.update_baseline_button = widgets.Button(description='Update Baseline')

        def on_update_baseline_button_clicked(b, self = self):

            if self.do_baseline:
                self.baseline_update()

        self.update_baseline_button.on_click(functools.partial(on_update_baseline_button_clicked, self=self))

        ##########################################

        # Add a button to save the plots to a pickle file
        self.save_data_button = widgets.Button(description='Save Baseline')

        def on_save_data_button_clicked(b, self = self):

            if self.output_filename[-4:] != '.pkl':
                self.output_filename += '.pkl'

            self.save_plot(self.output_filename)

        self.save_data_button.on_click(functools.partial(on_save_data_button_clicked, self=self))

        ##########################################

        # Add a text box to set the name of the output file
        self.save_name_text = widgets.Text(description='File Name',value=self.output_path)

        def on_save_name_text_entered(b, self = self):

            # Change output path
            self.output_filename = self.save_name_text.value

        self.save_name_text.observe(functools.partial(on_save_name_text_entered, self=self))

        ##########################################

    # Update the buttons onscreen
    def place_buttons(self):
        # Place the buttons and menus into a box and display it
        if self.button_group == 0:
            self.menu.children = [self.on_off_button,self.menu_button,self.add_button,self.move_button,self.delete_button,self.clear_all_button,self.hide_marker_button]
        elif self.button_group == 1:
            if self.fit_type == 'Polynomial':
                self.menu.children = [self.on_off_button,self.menu_button,self.update_baseline_button,self.fit_type_select,self.fit_order_inttext]
            else:
                self.menu.children = [self.on_off_button,self.menu_button,self.update_baseline_button,self.fit_type_select]
        elif self.button_group == 2:
            self.menu.children = [self.on_off_button,self.menu_button,self.scale_button,self.color_target_select,self.color_picker]
        elif self.button_group == 3:
            self.menu.children = [self.on_off_button,self.menu_button,self.save_name_text,self.save_data_button]
        # Set the output
        output = widgets.Output()
        # Display the box
        display(self.menu, output)
        self.fig.canvas.draw_idle()

    def clearfit(self):
        self.clear(self.plotfit)

    ##########################################
    ## FITTING METHODS
    ##########################################

    # Global fit function
    def fit(self):
        # If there is only one point, don't fit
        if len(self.xdat) < 2: return
        if self.fit_type == 'Polynomial': self.pfit()
        elif self.fit_type == 'Linear Segment': self.lsfit()

        # Update the baseline
        self.baseline_update()

    # Polynomial fit function
    def pfit(self):
        # Set the fit order to one below the number of points
        self.fit_order = np.min([5,len(self.xdat)-1])
        # If a fit order override is specified, override the fit order
        if self.fit_order_override != 0: self.fit_order = self.fit_order_override
        # Calculate a polynomial fit
        z = np.polyfit(self.xdat, self.ydat, self.fit_order)
        self.fitfunc = np.poly1d(z)

        # Calculate the fit arrays
        xnew = np.linspace(self.xbound_min, self.xbound_max, 200)
        ynew = self.fitfunc(xnew)

        # Plot the fit
        self.plotfit.set_data([xnew,ynew])

        return(xnew,ynew)

    # Linear segmented fit function
    def lsfit(self):

        # Calculate ordered copies of xdat and ydat
        xord,yord = self.order()
        # Create the fitting function
        self.fitfunc = interp1d(xord,yord)
        # Create x and y data using the fitting function
        x2 = np.linspace(np.min(xord),np.max(xord),500)
        y2 = self.fitfunc(x2)

        # Plot the fit
        self.plotfit.set_data([x2,y2])

        return

    ##########################################
    ## UPDATE METHODS
    ##########################################

    # Update the baseline
    def baseline_update(self):
        # If there is no baseline, return
        if not self.do_baseline: return
        # If there is no fit, return
        if len(self.xdat) == 0: return
        # Subset the x range to what is visible
        xord,yord = self.order()

        ii = np.squeeze(np.where((self.bsx > xord[0]) & (self.bsx < xord[-1])))
        self.sub_bsx = self.bsx[ii]
        self.sub_bsy = self.bsy[ii]
        # Calculate the fit using the baseline x data
        self.bs_fity = self.fitfunc(self.sub_bsx)
        # Show the residual plot
        self.bsplot.set_data([self.sub_bsx,self.sub_bsy - self.bs_fity])
        # Autoscale the y axis correspondingly
        self.bsax.relim()
        self.bsax.autoscale()

        self.int.intdata = [np.array(self.sub_bsx),np.array(self.sub_bsy - self.bs_fity)]
        self.int.fit()

    ##########################################
    ## RETURN METHODS
    ##########################################

    # Return the fit itself
    def get_fit(self):
        return self.fitfunc

    # Return the baseline x and y values
    def get_baseline(self):
        return self.sub_bsy - self.bs_fity

    # Return plot dictionary
    def get_plots(self,fig=False):
        # Initialize a dictionary to return
        plot_dict = {}
        # Input data to be baselined
        # (can i get this info from the fig or ax objects?)
        plots = [self.points,self.plotfit,self.bsplot,self.int.points,self.int.plotfit,self.int.fill]
        plotnames = ['points','fit','blined_curve','intpoints','intfit','intfill']

        for ind,plot in enumerate(plots):
            plot_info = {}

            # Save all relavant info to a dictionary
            if 'PolyCollection' in repr(type(plot)):
                try:
                    x,y = zip(*self.int.fill.get_paths()[0].vertices)
                    plot_info['data'] = [np.array(x),np.array(y)]
                    plot_info['color'] = self.fill_color
                except:
                    pass
            else:
                x,y = plot.get_data()
                plot_info['data'] = [np.array(x),np.array(y)]
                plot_info['color'] = plot.get_color()
                plot_info['marker'] = plot.get_marker()
                plot_info['markersize'] = plot.get_markersize()

            # Add the plot dictionary to the master dictionary
            plot_dict[plotnames[ind]] = plot_info
        # Add other useful data
        if fig:
            plot_dict['plots'] = plots
            plot_dict['fig'] = self.fig

        return plot_dict

    def save_plot(self,path,fig=False):
        # Retrieve plot info
        dict = self.get_plots(fig=fig)
        # Save to a pickle file
        pickle.dump(dict,open(path,'wb'))

class integrate(interactive):
    '''
    Integration Class
    '''
    # Initialize the class
    def __init__(self,fig=None,ax=None,x=[],y=[],intdata=None,hide_buttons=False):

        # Save whether or not to hide buttons
        self.hide_buttons = hide_buttons

        # Call the parent constructor
        super().__init__(fig=fig,ax=ax,x=x,y=y,menu_count=3)

        # Override axis scale, setting it to linear by default
        self.axscale = 1
        # Begin in integration mode by default
        self.do_integration = True
        self.do_gfit = False
        self.gnumber = 1

        # Initialize integration if intdata is provided
        if intdata != None:
            self.intdata = intdata
            self.fill = self.ax.fill_between([],[],[],color='skyblue',alpha=0.6)

        # Initialize the fit plot
        self.plotfit, = self.ax.plot([],[],color=self.fit_color)
        # Initialize the fit plot
        self.modfit, = self.ax.plot([],[],color='r')
        # BAD FORM: ASSUME THE FIRST PLOTTED LINE IS THE DATA
        self.plot = self.ax.lines[0]
        # Initialize the list of gaussian fits
        self.gfitlist = []

        self.ax.set_yscale('linear')

        # Plot the fit
        self.fit()

    # Define all buttons and their functionality
    def setup_custom_buttons(self):

        # Add a button widget to clear all points
        self.clear_all_button = widgets.Button(description='Clear All')

        def on_clear_all_button_clicked(b, self = self):

            # Clear all other points
            self.clear(self.newpoint)
            self.clear(self.selpoint)
            self.clear(self.modfit)
            [line.set_paths([]) for line in self.gfitlist if len(line.get_paths()[0]) > 0]
            self.selected = False

            # Clear all points
            self.xdat = np.array([])
            self.ydat = np.array([])

            # Replot all the points, excluding the selected one
            self.points.set_data([self.xdat,self.ydat])

            # Clear the fit
            self.plotfit.set_data([],[])

            # Replot the data
            self.plot.set_data(self.intdata[0],self.intdata[1])

            if self.do_integration:
                self.fill.set_paths([])

        self.clear_all_button.on_click(functools.partial(on_clear_all_button_clicked, self=self))

        ##########################################
        ## COLORS
        ##########################################

        # Add a color picker to allow custom colors
        self.color_picker = widgets.ColorPicker(description='Edit Color',concise=True,value=self.primary_color)

        def on_color_picker_clicked(b, self = self):

            # Save the user-selected color
            self.color = self.color_picker.value

            # If primary markers are selected
            if self.color_target == 'Marker - Primary':

                # Set the primary marker color and update the figure
                self.primary_color = self.color
                self.points.set_color(self.color)

            # If secondary markers are selected
            elif self.color_target == 'Marker - Secondary':

                # Set the secondary marker color
                self.secondary_color = self.color

            # If new markers are selected
            elif self.color_target == 'Marker - New':

                # Set the new marker color
                self.new_color = self.color

        self.color_picker.observe(on_color_picker_clicked)

        ##########################################
        ## Integration
        ##########################################

        # Add a button widget to add new points
        self.integrate_button = widgets.ToggleButton(description='Integrate',value=True)

        def on_integrate_button_clicked(b, self = self):

            if self.do_integration:
                self.do_integration = False
                self.fill.set_paths([])

            else:
                self.do_integration = True
                self.fit()
                if self.do_gfit:
                    # Reset any possible gaussian fitting
                    self.clear(self.modfit)
                    [line.set_paths([]) for line in self.gfitlist if len(line.get_paths()[0]) > 0]

                    # Shift all curves back down
                    self.shift(-1*self.yshift,self.stats,reset_data=True)
                    self.gfitlist = []

                    # Autoscale the y axis correspondingly
                    self.ax.relim()
                    self.ax.autoscale()

                    self.do_gfit = False
                    self.fit_model_button.value = False

        self.integrate_button.observe(functools.partial(on_integrate_button_clicked,self=self))#,'value')

        ##########################################
        ## Model Fitting
        ##########################################

        # Add a button widget to add new points
        self.fit_model_button = widgets.ToggleButton(description='Model Fit',value=False)

        def on_fit_model_button_clicked(b, self = self):

            if self.do_gfit:
                self.do_gfit = False

                # Clear the model and the underlying fits
                self.clear(self.modfit)
                [line.set_paths([]) for line in self.gfitlist if len(line.get_paths()[0]) > 0]

                # Shift all curves back down
                self.shift(-1*self.yshift,self.stats,reset_data=True)
                self.gfitlist = []

                # Autoscale the y axis correspondingly
                self.ax.relim()

            else:
                self.do_gfit = True
                # Calculate and plot the gaussian fits
                self.stats,self.yshift = self.gfit_plotting()
                self.shift(self.yshift,self.stats)
                if self.do_integration:
                    self.fill.set_paths([])
                    self.integrate_button.value = False
                    self.do_integration = False

        self.fit_model_button.observe(functools.partial(on_fit_model_button_clicked,self=self))

        ##########################################

        # Add a text box to set the fit order
        self.fit_number_inttext = widgets.IntText(description='Fit Number',value=1)

        def on_fit_number_inttext_entered(b, self = self):

            # Change fit order value
            self.gnumber = self.fit_number_inttext.value

        self.fit_number_inttext.observe(functools.partial(on_fit_number_inttext_entered, self=self))

        ##########################################

        # Add a button widget to add new points
        self.show_peaks_button = widgets.ToggleButton(description='Find Peaks',value=False)

        def on_show_peaks_button_clicked(b, self = self):

            xmin,xmax = self.ax.get_xlim()
            ymin,ymax = self.ax.get_ylim()

            specx,specy = self.intdata

            fspec = self.flatten(specy)
            vlines = self.find_spec_peaks(fspec)

            self.vlines = self.ax.vlines(specx[vlines],ymin,ymax,color='r')


        self.show_peaks_button.observe(functools.partial(on_show_peaks_button_clicked,self=self))

        ##########################################

    # Update the buttons onscreen
    def place_buttons(self):
        # If instructed to hide buttons
        if self.hide_buttons: return
        # Place the buttons and menus into a box and display it
        if self.button_group == 0:
            self.menu.children = [self.on_off_button,self.menu_button,self.add_button,self.move_button,self.delete_button,self.clear_all_button,self.hide_marker_button]
        elif self.button_group == 1:
            self.menu.children = [self.on_off_button,self.menu_button,self.integrate_button,self.fit_model_button,self.fit_number_inttext,self.show_peaks_button]
        elif self.button_group == 2:
            self.menu.children = [self.on_off_button,self.menu_button,self.scale_button,self.color_target_select,self.color_picker]
        # Set the output
        output = widgets.Output()
        # Display the box
        display(self.menu, output)
        self.fig.canvas.draw_idle()

    ##########################################
    ## FITTING METHODS
    ##########################################

    # Global fit function
    def fit(self):
        # If there is only one point, don't fit
        if len(self.xdat) < 2: return

        # Calculate ordered copies of xdat and ydat
        xord,yord = self.order()
        # Create the fitting function
        self.fitfunc = interp1d(xord,yord)
        # Create x and y data using the fitting function
        x2 = np.linspace(np.min(xord),np.max(xord),500)
        y2 = self.fitfunc(x2)

        # Plot the fit
        self.plotfit.set_data([x2,y2])

        if self.do_integration:
            self.fill.set_paths([])
            self.integration()

    def integration(self):

        # Calculate ordered copies of xdat and ydat
        xord,yord = self.order()

        # Find the min and max of the x array
        xmin,xmax = np.nanmin(xord),np.nanmax(xord)

        # Pull out the data
        xspec,yspec = self.intdata
        # Find the indices over which to integrate
        self.ii = np.squeeze(np.where((xspec > xmin) & (xspec < xmax)))
        ii = self.ii
        # Use scipy's simpsons integration technique -- negative for decreasing x range
        full_area = np.abs(scpyintegrate.simps(yspec[ii],xspec[ii]))

        # Calculate the slope and intercepts for the baseline
        slope = (yspec[ii[0]]-yspec[ii[-1]])/(xspec[ii[0]]-xspec[ii[-1]])
        yint = yspec[ii[0]] - slope * xspec[ii[0]]

        # Calculate the baseline values and integrate them
        baseline=slope * xspec[ii] + yint
        baseline_area = np.abs(scpyintegrate.simps(baseline,xspec[ii]))

        # Correct the full area by removing the baseline area
        self.area = full_area - baseline_area

        self.fill = self.ax.fill_between(xspec[ii],yspec[ii],self.fitfunc(xspec[ii]),color=self.fill_color,alpha=0.6)

    def clearfit(self):
        self.clear(self.plotfit)
        self.fill.set_paths([])

    def gfit_plotting(self,palette='Set2'):

        gnum = self.gnumber
        xspec,yspec = self.intdata

        xlimmin,xlimmax = self.ax.get_xlim()
        ylimmin,ylimmax = self.ax.get_ylim()

        stats = self.gfit(self.intdata,xlimmin,xlimmax,ylimmax,gnum)
        self.stats = stats

        area = np.round(sum(stats['area'].values()),4)

        # Best-fit model baseline corrected
        self.modfit, = self.ax.plot(xspec, stats['sim_corr'] + ylimmax,  color = 'red',lw = 2, alpha = 0.6, label='Model')

        # Individual Gaussian Components
        popt = stats['popt']
        colors = sns.color_palette(palette).as_hex()

        xrange = np.linspace(xlimmin,xlimmax,len(stats['xfit']))
        for i in range(0,gnum):
            line = self.ax.fill_between(xrange, self.gaussian(xrange, popt[3*i + 3], popt[3*i + 3 + 1], popt[3*i + 3 + 2]), color = colors[i%6], lw = 1, label = f'Model {i+1}',alpha=0.6)
            self.gfitlist.append(line)

        # self.ax.legend(fancybox=True, shadow=True)

        return(stats,ylimmax)

    ##########################################
    ## GAUSSIAN FITTING SUPPORT METHODS
    ##########################################

    def gaussian(self,xx,sigma,a,mu):
        # A gaussian function used for curve fitting
        exp = -(1/2) * ((xx-mu)/sigma) ** 2
        return a * np.exp(exp)

    # Define an n gaussian model
    def model(self,xx, slope, yoff, xoff, *params):
        y = np.zeros_like(xx)
        for i in range(0,len(params),3):
            y += self.gaussian(xx,params[i],params[i+1],params[i+2])
        return(y + yoff + slope * (xx-xoff))

    # Method to determine guess parameters and bounds
    def get_params(self,xmin,xmax,ymax,gnum=1):
        #Set initial parameters and bounds

        xavg = (xmin+xmax)/2

        #             slope   yoff    xoff
        p0      =  [ -1e-6,  -1e-3,   xavg ]   # GUESSES
        bound1  =  [ -1,     -1,     -1e4  ]   # LOWER BOUNDS
        bound2  =  [  1,      1,      1e4  ]   # UPPER BOUNDS

        #             std  a      mu
        p0_fit   =  [ 2,   2e-2,  xavg ]   # GUESSES
        bnd_fit1 =  [ 0,   1e-6,  xmin ]  # LOWER BOUNDS
        bnd_fit2 =  [ 5,   ymax,  xmax ]  # UPPER BOUNDS

        # Combine the separate parameter arrays
        for i in range(gnum):
            p0 = np.concatenate((p0,p0_fit))
            bound1 = np.concatenate((bound1,bnd_fit1))
            bound2 = np.concatenate((bound2,bnd_fit2))
        bounds = np.concatenate(([bound1],[bound2]),axis=0)

        # Return the initial parameter guesses and bounds
        return(p0,bounds)

    # Method to return stats about a certain fit
    def fit_stats(self,stats,gnum):

        area,area_err,fwhm,fwhm_err = {},{},{},{}
        popt,perr = stats['popt'],stats['perr']

        # Calculate the area of a Gaussian (for example, if you want to calculate the column density)
        for i in range(0,gnum):
            # area of the first Gaussian: area = sqrt(2pi)*width*amp
            area[f'fit{i}'] = np.sqrt(2*np.pi)*popt[i*3 + 3]*popt[i*3 + 3 + 1]
            # error of the area of the first Gaussian calculated from the errors in the Gaussian parameters
            area_err[f'fit{i}_err'] = area[f'fit{i}'] * np.sqrt((perr[i*3 + 3]/popt[i*3 + 3])**2 + (perr[i*3 + 3 + 1]/popt[i*3 + 3 + 1])**2)

            # The FWHM (full width half maximum) of the Gaussian can be calculated from the width parameter of the Gaussian
            fwhm[f'fit{i}'] = 2.35482 * popt[3*i + 3]
            fwhm_err[f'fit{i}_err'] = 2.35482 * perr[3*i + 3]

        stats['area'],stats['area_err'],stats['fwhm'],stats['fwhm_err'] = area,area_err,fwhm,fwhm_err

        return(stats)

    # Fit gaussians to data
    def gfit(self,data,xmin,xmax,ymax,gnum=1):

        x,y = data

        p0,bounds = self.get_params(xmin,xmax,ymax,gnum)

        # Select the region of the spectrum that will be fitted with the model
        ii = np.squeeze(np.where((x > xmin) & (x < xmax)))

        xfit,yfit = x[ii],y[ii]

        # Call curve fit
        popt, pcov = curve_fit(self.model, xfit, yfit, p0=p0, bounds=bounds)

        # Fit products
        # The model only fits the baseline in the selected region, so the corrected spectrum could look weird outside that region
        # because we're using a local baseline for the whole spectrum

        bl = popt[1] + popt[0] * (x - popt[2])     #best-fit baseline
        y_corr  = y - bl                           #baseline-subtracted real data\n",

        simspec = self.model(x, *popt)                  # create a synthetic spectrum of your fit
        sim_corr = simspec - bl                    # baseline-subtracted synthetic spectrum",

        # pcov is a matrix with values related to the errors of the fit.
        # To get the actual errors of the Gaussian parameters one needs to calculate the square root of the values in the diagonal

        perr = np.sqrt(np.diag(pcov))              #calculate fit errors

        stats = {'popt':popt,'perr':perr,'xfit':xfit,'yfit':yfit,'y_corr':y_corr,'sim_corr':sim_corr,'simspec':simspec,'bl':bl}

        stats = self.fit_stats(stats,gnum)

        return(stats)

    def shift(self,yshift,stats,reset_data=False):

        x,y = self.points.get_data()
        self.points.set_data(x,np.array(y)+yshift)
        self.ydat = np.array(self.ydat) + yshift
        if reset_data: self.plot.set_data(self.intdata[0],self.intdata[1])
        else: self.plot.set_data(self.intdata[0],stats['y_corr'] + yshift)
        self.fit()

    ##########################################
    ## RETURN METHODS
    ##########################################

    def flatten(self,spec):
        normspec = spec/np.nanmax(spec)
        spec_medavg = medfilt(normspec,kernel_size=51)
        spec_peaks = np.abs(normspec - spec_medavg)
        return spec_peaks
    def find_spec_peaks(self,spec,height=0.002,distance=6):
        peaks,_ = find_peaks(spec,height=height,distance=distance)
        return peaks

    # # Return the plotted points
    # def get_points(self):
    #     return self.xdat,self.ydat

class spectrum(integrate):
    '''
    Spectrum handling class
    '''
    def __init__(self,fig=None,ax=None,x=[],y=[],spectrum=None,hide_buttons=False):
        super().__init__(fig=fig,ax=ax,x=x,y=y,intdata=spectrum,hide_buttons=hide_buttons)

    def setup_custom_buttons(self):

        ##########################################
        ## Integration
        ##########################################

        # Add a button widget to add new points
        self.integrate_button = widgets.ToggleButton(description='Integrate',value=True)

        def on_integrate_button_clicked(b, self = self):

            if self.do_integration:
                self.do_integration = False
                self.fill.remove()
            else:
                self.do_integration = True
                self.fit()

        self.integrate_button.observe(on_integrate_button_clicked,'value')


    def spec_peaks(spec,height=0.002,distance=6):
        normspec = spec/np.nanmax(spec)
        sp_medavg = medfilt(normspec,kernel_size=51)
        flat_spec = np.abs(normspec - sp_medavg)
        self.peaks,_ = find_peaks(flat_spec,height=height,distance=distance)

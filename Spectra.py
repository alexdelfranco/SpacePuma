# June 14, 2022
# Author: Alex DelFranco
# Advisor: Rafael Martin-Domenech
# Purpose: For Analyzing Ice Spectra

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import glob


################################################################################
# Import Methods
################################################################################

def load(datadir,date,interp_offset=0):
    '''
    Main load method to import all data into a data dictionary
    '''
    interp_offset = 0
    data = {} # Create a dict to hold all the data from the day
    folders = glob.glob(f'{datadir}/{date}/*')
    if f'{datadir}/{date}/Extra Data' in folders: folders.remove(f'{datadir}/{date}/Extra Data')

    # Loop through the data taken on that day
    for folder in folders: data[folder.split('/')[-1]],gen_data = folder_load(folder,interp_offset)
    gen_data['Date'] = date
    data['Info'] = gen_data
    return(data)

def path_extract(fname):
    '''
    Extracts all useful information from the file name and organizes the info into a dictionary
    '''
    file_data,gen_data = {},{}
    namedat = fname.split('-')
    # For each section of the file name, save the following infomration
    for datstr in namedat:
        info = datstr.split('_')
        if info[0] == 'gas': gen_data['Gas'] = info[1:]
        if info[0] == 'icetemp': gen_data['Icetemp'] = int(info[-1][:-1])
        if info[0] == 'dep':
            gen_data['Dep'] = float(info[1][:-3])
            file_data['Label'] = str(gen_data['Dep'])
        if info[0] == 'irr':
            file_data['Irr'] = info[1:]
            file_data['Time'] = info[1][:-3]
            file_data['Label'] = file_data['Time']
        if info[0] == 'temp':
            file_data['Temp'] = info[1]
            file_data['Label'] = str(file_data['Temp'])

        if info[0] == 'tpd':
            file_data['TPD'] = info[1:]
            file_data['Time'] = info[2][:-3]
            file_data['Temp'] = fname.split('.')[-1][:-1]

    return file_data,gen_data

def folder_load(folder_path,interp_offset=0):
    '''
    Load in one of the subfolders for a given day
    '''
    fdata = {} # Create a new dictionary for data from the specified folder
    flist = glob.glob(f'{folder_path}/*')
    # For each data file with .dpt endings
    dpt_dict = {}
    for file in flist:
        if '.dpt' in file:
            if 'bkg' in file: continue
            load_data,gen_data = dpt_load(file)
            if 'TPD' in folder_path: lab = load_data['Temp']
            if 'Irradiation' in folder_path: lab = load_data['Label']
            dpt_dict[lab] = load_data
            fdata['Spectra'] = dpt_dict
            # fdata['Info'] = gen_data

        #Read in excel file with temperature data into dataframe
        elif '.xls' in file: tpd_temp = pd.read_excel(file, index_col=None, header=3)
        elif '.asc' in file:
            # Read the mass data into a dataframe
            tpd_qms = tpd_readqms(file)
            fdata['TPD_qms'] = tpd_qms
    if 'TPD' in folder_path:
        if '20220711' in folder_path:
            tpd_qms = tpd_qms.iloc[3055:,:]
            fdata['TPD_qms'] = tpd_qms
            temp_int = np.arange(8,190,182/len(tpd_qms['time']))
        else:
            tpd_interp = interp1d(round(tpd_temp.loc[:,'Time']*1.e-3+interp_offset), tpd_temp.loc[:,'Input B'], kind='cubic',fill_value='extrapolate')  #Interpolate time+temperature data
            temp_int = tpd_interp(fdata['TPD_qms'].loc[:,'time'])   # Assign interplated temperature values to the tpd data
        fdata['TPD_qms']['Temp'] = temp_int # Add the interpolated temperature values to the qms dataframe
    return fdata,gen_data


def dpt_load(file):
    '''
    General method to load dpt files
    '''
    data = {}
    file_name = file.split('/')[-1] # Find the file name
    name = file_name.split('.dpt')[0]
    x,y = zip(*np.loadtxt(file))
    x,y = np.array(x),np.array(y)
    file_data,gen_data = path_extract(name)
    file_data['Name'],file_data['Wavenumber'],file_data['Absorption'] = name,x,y
    return file_data,gen_data


def tpd_readqms(file_name,print_masses=False):
    '''
    A function to read in raw QMS data
    '''
    count = len(open(file_name).readlines(  ))  #Extract number of lines in TPD file to later be able to ignore the last one
    h_len = 6  #Number of lines of information before the list of masses

    f = open(file_name, 'r')  #Open TPD data file for reading

    for j in range(h_len):  #Read the initial set of header data and discard
        header = f.readline()

    header_m = f.readline()  #Read header line with masses
    masses = header_m.split()  #Split header row into a list of masses (as strings)
    if print_masses: print(f'Measured Masses: {masses}')  #Print masses and check that they are correct

    line = f.readline()  #Read line of the TPD file
    data = []  #Set up a data dictionary
    pos = 0  #Set a line position counter

    for line in f:
        pos += 1  #Increase position counter by 1
        columns = line.split()  #Split row into a list of strings
        source = {}  #Define a dictonary row
        source['time'] = float(columns[3])   #Add the 4th element as in the row as a float to source with label 'time'
        if pos < (count-h_len-3): #do not read the last line of the file which is often broken
            for i in range(len(masses)):  #for loop to assign the values of the different masses
                source[masses[i]] = float(columns[4+i*5])  #Add the 5th +i*5th element as floats in the row to the different masses
            data.append(source)  #Add dictionary row to data dictionary

    f.close()  #Close file for reading
    data = pd.DataFrame.from_dict(data)  #Convert dictionary to a pandas dataframe for easy manipulation

    return(data)

################################################################################
# Basic Plotting Methods
################################################################################

def specplot(data,mol_list=[],shift=0,lab_ext='',title='Spectra',notitle=False):
    '''
    Base plotting method for plotting spectra
    '''
    plt.close()
    fig,ax = plt.subplots(figsize=(12,6)) # Setup the figure
    order = [str(y) for y in sorted([int(x) for x in data.keys()])]
    shift_indx = np.squeeze(np.where(np.around(data[order[0]]['Wavenumber']) == 2500))[0]

    for key in order: # For each data list
        spectra = data[key]
        # Shift the data vertically
        full_shift = shift * order.index(key)
        # Plot the data on the axis
        ax.plot(spectra['Wavenumber'],spectra['Absorption']-spectra['Absorption'][shift_indx] + full_shift,label=f'{key}{lab_ext}')
    ax.set_xlabel('Wavenumber',size=18)
    ax.set_ylabel('Absorption',size=18)
    if not notitle: ax.set_title(title,size=22,font='Times',pad='10')
    xleft,xright = ax.get_xlim()
    if xleft < xright: ax.invert_xaxis()
    ax.legend(fancybox=True, shadow=True) # Plot a legend

    # Plot vertical lines at given points
    ymin,ymax = ax.get_ylim()
    for mol in mol_list:
        for line in mol_list[mol]:
            ax.vlines(line,ymin,ymax,'k','--',alpha=0.5)
    plt.show()

    return(fig,ax)

def irrplot(data_dict,mol_list=[],shift=0,dev=False):
    '''
    Method for plotting Irradiation Spectra
    '''
    fig,ax = specplot(data_dict['Irradiation']['Spectra'],mol_list,shift=shift,lab_ext=' Min',title='Irradiation Spectra')
    if dev: return(fig,ax)

def tpdplot(data_dict,mol_list=[],shift=0,dev=False,notitle=False):
    '''
    Method for plotting TPD Spectra
    '''
    temp = data_dict['Info']['Icetemp']
    fig,ax = specplot(data_dict['TPD']['Spectra'],mol_list,shift=shift,lab_ext=' K',title=f'TPD Spectra - {temp}K',notitle=notitle)
    if dev: return(fig,ax)

def tpd(data_dict,mass_list=range(0,100)):
    '''
    Method for plotting TPD Curves
    '''
    plt.close()
    fig,ax = plt.subplots(figsize=(12,7)) # Setup the figure
    ax.set_yscale('log')
    ax.set_title('Temperature Programmed Desorption',size=22)
    ax.set_xlabel('Temperature (K)',size=18)
    ax.set_ylabel('QMS Signal (log)',size=18)
    df = data_dict['TPD']['TPD_qms']

    for col in df.columns[1:-1]:
        if int(col) in mass_list:
            ax.plot(df['Temp'],df[col],label=f'Mass {col}')
    ax.legend()
    plt.show()

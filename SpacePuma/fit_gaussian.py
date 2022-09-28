# Author: Alex DelFranco
# Advisor: Rafa Martin Domenech
# Institution: Center for Astrophysics | Harvard & Smithsonian
# Date: 28 August 2022

class fit_gaussian(fit):
    def __init__(self,fig,menu=None,data=None,artists_global=None,data_global=None,load=None):


    ##########################################
    ## THE MODEL
    ##########################################

    def gaussian(self,xx,sigma,a,mu):
        # A gaussian function used for curve fitting
        exp = -(1/2) * ((xx-mu)/sigma) ** 2
        return np.array(a * np.exp(exp))

    # Define an n gaussian model
    def ngaussian_model(self,xx,*params):
        y = np.zeros_like(xx)
        for i in range(0,len(params),3):
            y = np.add(y, self.gaussian(xx,params[i],params[i+1],params[i+2]), casting="unsafe")
        # return(y + yoff + slope * (xx-xoff))
        return y

    ##########################################
    ## PARAMETERS
    ##########################################

    # Method to determine guess parameters and bounds
    def get_params(self,xmin,xmax,ymax,gnum=1):
        #Set initial parameters and bounds

        xavg = (xmin+xmax)/2
        std_max = self.info['std_max']

        p0,bound1,bound2 = [],[],[]

        #             std        a        mu
        p0_fit   =  [ std_max/2, ymax,  xavg ]   # GUESSES
        bnd_fit1 =  [ 0,         1e-6,    xmin ]  # LOWER BOUNDS
        bnd_fit2 =  [ std_max,   ymax*2, xmax ]  # UPPER BOUNDS

        # Combine the separate parameter arrays
        for i in range(gnum):
            p0 = np.concatenate((p0,p0_fit))
            bound1 = np.concatenate((bound1,bnd_fit1))
            bound2 = np.concatenate((bound2,bnd_fit2))
        bounds = np.concatenate(([bound1],[bound2]),axis=0)

        # Return the initial parameter guesses and bounds
        return(p0,bounds)

    ##########################################
    ## STATS
    ##########################################

    # Method to return stats about a certain fit
    def add_fit_stats(self,stats,gnum):

        area,area_err,fwhm,fwhm_err = {},{},{},{}
        popt,perr = stats['popt'],stats['perr']

        # Calculate the area of a Gaussian (for example, if you want to calculate the column density)
        for i in range(0,gnum):
            # area of the first Gaussian: area = sqrt(2pi)*width*amp
            area[f'fit{i+1}'] = np.sqrt(2*np.pi)*popt[i*3]*popt[i*3 + 1]
            # error of the area of the first Gaussian calculated from the errors in the Gaussian parameters
            area_err[f'fit{i+1}_err'] = area[f'fit{i+1}'] * np.sqrt((perr[i*3]/popt[i*3])**2 + (perr[i*3 + 1]/popt[i*3 + 1])**2)

            # The FWHM (full width half maximum) of the Gaussian can be calculated from the width parameter of the Gaussian
            fwhm[f'fit{i+1}'] = 2.35482 * popt[3*i]
            fwhm_err[f'fit{i+1}_err'] = 2.35482 * perr[3*i]

        stats['area'],stats['area_err'],stats['fwhm'],stats['fwhm_err'] = area,area_err,fwhm,fwhm_err

        return(stats)

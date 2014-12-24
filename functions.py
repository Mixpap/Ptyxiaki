from astropy.io import fits
from astropy import wcs
import numpy as np
import pywcsgrid2
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ab12CO = 77.0
ab13CO = 8.0

def open_map(fits_file):
    """
    Usage: Pixel_map, Wcs_Coords = open_map('fits_file')
    Input: A 2D/3D fits file
    Return: Numpy Masked Array of pixel Map,
            WCS object of Map
    """
    image = fits.getdata(fits_file)
    image= np.ma.masked_array(image , np.isnan(image))
    w = wcs.WCS(fits_file)
    return image,w

def denoise_map(m,rms):
    """
    Usage: DeNoised_map = denoise_map(map,RMS_map)
    Input: A 2D pixel map
    Return: Numpy Masked Array where snr>3
    """
    return np.ma.masked_where(m<3.0*rms,m)

def cube_to_max(cube):
    """
    Input: A 3D Masked Cube_Map
    Return: Masked Maximum Intensity Map
    """
    X=cube.shape[2]
    Y=cube.shape[1]
    mapmax=np.ma.masked_array(np.zeros((Y,X)),mask=np.ones((Y,X)))
    for y in range(Y):
        for x in range(X):
            value = cube[:,y,x].max()
            if (value.dtype=='float32'):
                mapmax[y,x]=value
    return mapmax

def plot_map(image,coords,title='title',cm ='RdPu'):
    """
    Usage: plot_map(pixel_map,WCS_Coords)
    Input: 2D pixel Map, WCS object of the Map
    Output: Plot of The Image with a WCS Compass
    """
    ax1 = pywcsgrid2.subplot(111, wcs=coords)
    im = ax1.imshow(image,origin='low',cmap=cm)
    ax1.add_compass(loc=5,color='black')
    ax1.set_title(title)
    plt.colorbar(im)

def plot_maps(coords,cm='RdPu',**kwargs):
    """
    Usage: plot_maps(maps=[list_of_maps],titles=[list_of_titles],coords=wcs,cm=colormap)
    Input: 2D pixel Maps, WCS object of the Maps
    Output: Plot all Subplots of The Images with WCS Compass
    """
    nm=len(kwargs['maps'])
    a=[[]]*nm
    for i,m in enumerate(kwargs['maps']):
        vmax=m.max()
        a[i]=(pywcsgrid2.subplot(1,nm,i,wcs=coords))
        im=a[i].imshow(m,origin='low',cmap=cm,norm=LogNorm())
        a[i].set_title(kwargs['titles'][i])
        a[i].add_compass(loc=5,color='black')
        try:
            plt.colorbar(im, fraction=0.040, pad=0.04,ticks=np.logspace(-1,np.log10(vmax),10),format="%.1f")
        except:
            pass
        plt.tight_layout()

def initial_est(map_thick,map_thin,abundance):
    """
    Initial Estimation of Optical Thickness using
    two Isotopes and its abundance ratio
    Input: Pixel_map of one isotope, Pixel_map of Optical Thick Isotope,
    abudance ratio
    Return: Masked Pixel_map of Optical Thickness
    """
    map_thick=np.ma.masked_where(map_thick==0.0,map_thick)
    ratio = map_thin/map_thick
    tau=-abundance*np.log(1-ratio)
    return tau

def tau_new(tau,map_thick,map_thin,abundance):
    ratio=map_thick/map_thin
    e=np.exp(-tau)
    eab=np.exp(-tau/abundance)
    return tau-(ratio*(1-eab)-(1-e))/(ratio*eab/abundance-e)

def final_est(map_thick,map_thin,T0,abundance,maxiter=5):
    """
    Final Estimation of Optical Thickness using two Isotopes,
    its abundance ratio and the Initial Estimation
    Input: Pixel_map of Optical Thick isotope, Pixel_map of Optical Thin Isotope,
    Initial estimation Pixel_map, abudance ratio and MaxIterations = 5
    Return: Pixel_map of Optical Thickness
    """
    for i in range(maxiter):
        tau=tau_new(T0,map_thick,map_thin,abundance)
    return tau

def Tr_est(Ta,nfss=0.77):
    """
    True Temperature Estimation
    Input: Pixel_map of Optical Thick Isotope, fss parameter
    Return: Pixel_map of True Temperature
    """
    return Ta/nfss

def gaussian(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def gauss_fit(T,N):
    """
    !! Warning: Extreme Slow !!
    Runs a Scipy Curve Gaussian fit through every pixel of non-Masked Map
    Input: A 3D pixel map
    Return: 2D Map of GPeak Position, 2D Map of FWHM
    """
    #Z,Y,X=T.shape[0],N,N
    Z,Y,X=T.shape[0],T.shape[1],T.shape[2]
    xx=np.linspace(0,1,61)
    Peak_Map=np.zeros((Y,X))
    HW=np.zeros((Y,X))
    for y in range(Y):
        for x in range(X):
            s=Spectra(T,y,x)
            if np.ma.is_masked(s):
                Peak_Map[y,x]=0.0
                FW[y,x]=0.0
            else:
                try:
                    popt, pcov = curve_fit(gaussian,xx,s)
                    Peak_Map[y,x]=popt[1]
                    FW[y,x]=2.355*np.abs(popt[2])
                except:
                    Peak_Map[y,x]=0.0
                    FW[y,x]=0.0
                    pass
    return Peak_Map,FW


def Tx_est(v,Tr,Tbg=2.7):
    """
    Excitation Temperature Estimation
    Input: Frequency of observation in Hz, Pixel_map of Optical Thick Isotope, Cosmic Background Temperature
    Return: Pixel_map of Excitation Temperature
    """
    T0=0.04535*v
    A=Tr+T0*(np.exp(T0/Tbg)-1)
    Tx = T0/np.log(1+T0/A)
    return Tx
    # Y,X=Tr.shape[0],Tr.shape[1]
    # Tx=Tr.copy()
    # T0=0.04535*v
    # print T0
    # print T0*(np.exp(T0/Tbg)-1)
    # for y in range(Y):
    #     for x in range(X):
    #         if np.isnan(Tr[y,x]):
    #             Tx[y,x]=np.nan
    #         else:
    #             A=Tr[y,x]+T0*(np.exp(T0/Tbg)-1)
    #             Tx[y,x]=T0/np.log(1+T0/A)


def Spectra(cube_map,y,x):
    """
    Spectrum of Selected Coordinates
    Input: Cube Map, pixel Coordinates
    Return: Vector of Spectrum Values
    """
    return cube_map[:,y,x]

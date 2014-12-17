from astropy.io import fits
from astropy import wcs
import numpy as np
import pywcsgrid2
import matplotlib.pyplot as plt

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

def plot_map(image,coords):
    """
    Usage: plot_map(pixel_map,WCS_Coords)
    Input: 2D pixel Map, WCS object of the Map
    Output: Plot of The Image with a WCS Compass
    """
    ax1 = pywcsgrid2.subplot(111, wcs=coords)
    ax1.imshow(image,origin='low',cmap='jet')
    ax1.add_compass(loc=4,color='black')
    plt.colorbar()

def plot_maps(map1,w1,map2,w2):
    """
    Usage: plot_maps(pixel_map1,WCS_Coords_for_map1,pixel_map2,WCS_Coords_for_map2)
    Input: 2D pixel Maps, WCS object of the Maps
    Output: Plot two Subplots of The Images with WCS Compass
    """
    ax1 = pywcsgrid2.subplot(221, wcs=w1)
    ax1.imshow(map1,origin='low',cmap='gray')
    #ax1.set_title('$^{12}CO$ (max)')
    ax1.add_compass(loc=4,color='black')
    ax1.grid()

    ax2 = pywcsgrid2.subplot(222, wcs=w2)
    ax2.imshow(map2,origin='low',cmap='gray')
    #ax2.set_title('$C^{18}O$ (max)')
    ax2.add_compass(loc=4,color='black')
    ax2.grid()

def initial_est(map_thick,map_thin,abundance):
    """
    Initial Estimation of Optical Thickness using
    two Isotopes and its abundance ratio
    Input: Pixel_map of one isotope, Pixel_map of Optical Thick Isotope,
    abudance ratio
    Return: Masked Pixel_map of Optical Thickness
    """
    map_thick=np.ma.masked_where(map_thick==0.0,map_thick)
    ratio = map_thick/map_thin
    tau=-abundance*np.log(1-ratio)
    return tau

def tau_new(tau,map_thick,map_thin,abundance):
    ratio=map_thin/map_thick
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
        tau=tau_new(tau,map_thick,map_thin,abundance)

def Tr_est(Ta,nfss=0.77):
    """
    True Temperature Estimation
    Input: Pixel_map of Optical Thick Isotope, fss parameter
    Return: Pixel_map of True Temperature
    """
    return Tr/nfss

def Tx_est(v,Tr,Tbg=2.7):
    """
    Excitation Temperature Estimation
    Input: Frequency of observation in GHz, Pixel_map of Optical Thick Isotope, Cosmic Background Temperature
    Return: Pixel_map of Excitation Temperature
    """
    Y,X=Tr.shape[0],Tr.shape[1]
    Tx=Tr.copy()
    T0=0.04535*v
    print T0
    print T0*(np.exp(T0/Tbg)-1)
    for y in range(Y):
        for x in range(X):
            if np.isnan(Tr[y,x]):
                Tx[y,x]=np.nan
            else:
                A=Tr[y,x]+T0*(np.exp(T0/Tbg)-1)
                Tx[y,x]=T0/np.log(1+T0/A)
    return Tx


def thin_est(T):
    """
    Optical Thin Map
    Input: Pixel_map of Optical Thickness
    Return: Pixel_map of Optical Thin regions (T<1)
    """
    Y,X=T.shape[0],T.shape[1]
    Thin=T.copy()
    for y in range(Y):
        for x in range(X):
            if Thin[y,x]>1:
                Thin[y,x]=np.nan
    return Thin


def Spectra(cube_map,y,x):
    """
    Spectrum of Selected Coordinates
    Input: Cube Map, pixel Coordinates
    Return: Vector of Spectrum Values
    """
    return cube_map[:,y,x]

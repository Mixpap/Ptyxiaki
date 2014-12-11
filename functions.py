from astropy.io import fits
from astropy import wcs
import numpy as np
import pywcsgrid2
import matplotlib.pyplot as plt

ab12CO = 66.0
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



def initial_est(map1,map_thick,abundance):
    """
    Initial Estimation of Optical Thickness using
    two Isotopes and its abundance ratio
    Input: Pixel_map of one isotope, Pixel_map of Optical Thick Isotope,
    abudance ratio
    Return: Masked Pixel_map of Optical Thickness
    """
    #T0=np.zeros(map1.shape)
    T0=np.ma.masked_array(np.zeros((map1.shape)),mask=np.zeros((map1.shape)))
    Y,X=T0.shape[0],T0.shape[1]
    ab = abundance
    for y in range(Y):
        for x in range(X):
            if (np.isnan(map1[y,x]) or np.isnan(map_thick[y,x]) or (map_thick[y,x]==0.0) or (map1[y,x] >= map_thick[y,x]) or (map1.mask[y,x])or (map_thick.mask[y,x])):
                        T0.mask[y,x]=True
            else:
                T0[y,x]=-ab*np.log(1-map1[y,x]/map_thick[y,x])
    return T0


def final_est(map1,map_thin,T0,abundance):
    """
    Final Estimation of Optical Thickness using two Isotopes,
    its abundance ratio and the Initial Estimation
    Input: Pixel_map of one isotope, Pixel_map of Optical Thick Isotope,
    Initial estimation Pixel_map and abudance ratio
    Return: Pixel_map of Optical Thickness, Vector with Convergence Numbers per loop
    """
    Y,X=T0.shape[0],T0.shape[1]
    T=T0.copy()
    loops=[]
    ab=abundance
    tol=0.0001
    maxi=20
    for y in range(Y):
        for x in range(X):
            if (np.isnan(T[y,x]) or (map1[y,x]==0.0) or (T.mask[y,x]) or (map1.mask[y,x]) or (map_thin.mask[y,x])):
                T.mask[y,x]=True
            else:
                i=0
                f=0.1
                while f>tol:
                    A=map_thin[y,x]/map1[y,x]
                    B=1.0-np.exp(-T[y,x]/ab)
                    C=1.0-np.exp(-T[y,x])
                    D=np.exp(-T[y,x]/ab)/ab
                    F=np.exp(-T[y,x])
                    temp=T[y,x]
                    T[y,x]=T[y,x]-(A*B-C)/(A*D-F)
                    f=T[y,x]-temp
                    i=i+1

                    if i>=maxi:
                        break
                loops.append(i)
    return T,loops

def Tr_est(Ta,nfss=0.77):
    """
    True Temperature Estimation
    Input: Pixel_map of Optical Thick Isotope, fss parameter
    Return: Pixel_map of True Temperature
    """
    Y,X=Ta.shape[0],Ta.shape[1]
    Tr=Ta.copy()
    for y in range(Y):
        for x in range(X):
            Tr[y,x]=Ta[y,x]/nfss
    return Tr

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

def plot_noisy(map1,y,x,noise):
    csnr=map1/noise
    cp=np.ma.masked_where(csnr<3,csnr)

    ax1 = plt.subplot(221)
    ax1.imshow(map1,origin='low')
    ax1.set_title('Original Map')
    ax1.axvline(x,color='r')
    ax1.axhline(y,color='r')

    ax2 = plt.subplot(224)
    ax2.plot(map1[:,x])
    ax2.axhline(3*noise,color='r')
    ax2.axvline(y,color='k')
    ax2.set_xlabel('Y')

    ax3 = plt.subplot(223)
    ax3.plot(map1[y,:])
    ax3.axhline(3*noise,color='r')
    ax3.axvline(x,color='k')
    ax3.set_xlabel('X')

    ax4 = plt.subplot(222)
    ax4.imshow(cp,origin='low')
    ax4.set_title('Denoised Map')
    ax4.axvline(x,color='r')
    ax4.axhline(y,color='r')

def Spectra(cube_map,y,x):
    """
    Spectrum of Selected Coordinates
    Input: Cube Map, pixel Coordinates
    Return: Vector of Spectrum Values
    """
    return cube_map[:,y,x]

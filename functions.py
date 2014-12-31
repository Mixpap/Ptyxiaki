from astropy.io import fits
from astropy import wcs
import numpy as np
import pywcsgrid2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
ab12CO = 77.0
ab13CO = 8.0
hk=0.0479924335     #* 10e9
v_co12_j32=345.796  #GHz
v_co13_j32=330.5879 #GHz
v_c18o_j32=329.331  #GHz

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

def plot_maps(coords,cm='RdPu',norm='log',**kwargs):
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
        if norm=='log':
            im=a[i].imshow(m,origin='low',cmap=cm,norm=LogNorm())
        else:
            im=a[i].imshow(m,origin='low',cmap=cm)
        a[i].set_title(kwargs['titles'][i])
        a[i].add_compass(loc=5,color='black')
        try:
            if norm=='log':
                plt.colorbar(im, fraction=0.040, pad=0.04,ticks=np.logspace(-1,np.log10(vmax),10),format="%.1f")
            else:
                plt.colorbar(im,fraction=0.040, pad=0.04)
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
    T0=hk*v
    A=Tr+T0/(np.exp(T0/Tbg)-1)
    Tx = T0/np.log(1+A)
    return Tx


def Spectra(cube_map,y,x):
    """
    Spectrum of Selected Coordinates
    Input: Cube Map, pixel Coordinates
    Return: Vector of Spectrum Values
    """
    return cube_map[:,y,x]


def map_showXY(map12,map12m,map13,map18,ta12,ta13,wcs,y,x):
    """
    To use with IPython interact
    """
    #T=CO12[line,::]
    #ym,xm=(y,x)
    zlen=61.
    xx=np.linspace(0,1,zlen)
    #===========CO12=======================
    s12=Spectra(map12,y,x)
    der1=np.diff(s12)
    der2=np.diff(der1)
    m = np.ones(len(s12), dtype=bool)
    ind1=np.argsort(der2)[0]+1
    ind2=np.argsort(der2)[1]+1
    if np.abs(ind1-ind2)<2:
        ind2=np.argsort(der2)[2]+1
    m[ind1:ind2]=False
    m[ind2:ind1]=False
    popt12, pcov12 = curve_fit(gaussian, xx[m], s12[m],p0=[s12.max(),xx[np.argmax(s12)],0.1])
    sd12= np.sqrt(np.diag(pcov12))
    # if (sd12>popt12).all:
    #     popt12=np.zeros(3)
    #     sd12=np.zeros(3)

    FWHM12=2.355*np.abs(popt12[2])

    #===========CO13=======================
    s13=Spectra(map13,y,x)
    #max13 = xx[np.argmax(s13)] #trik
    #gn=gaussian(xx,1.,max13,2.5*popt12[2]) #trik
    #s13f=s13*gn #trik
    popt13, pcov13 = curve_fit(gaussian, xx, s13,p0=[0.25*popt12[0],popt12[1],popt12[2]])
    sd13= np.sqrt(np.diag(pcov13))
    # if (sd13>popt13).all:
    #     popt13=np.zeros(3)
    #     sd13=np.zeros(3)

    FWHM13=2.355*np.abs(popt13[2])

    #===========CO18=======================
    s18=Spectra(map18,y,x)
    #max18 = xx[np.argmax(s18)] #trik
    #gn=gaussian(xx,1.,max18,1.2*popt12[2]) #trik
    #s18f=s18*gn #trik
    #=========================================
    popt18, pcov18 = curve_fit(gaussian, xx, s18,p0=[0.25*popt13[0],popt13[1],popt13[2]])
    sd18= np.sqrt(np.diag(pcov18))

    FWHM18=2.355*np.abs(popt18[2])

    #================Optical--Thickness==========
    print 'tau12: %0.3f'%ta12[y,x]
    print 'tau13: %0.3f'%ta13[y,x]

    #===========PLOTS=======================
    gs = gridspec.GridSpec(7, 5)
    #ax1 = pywcsgrid2.subplot(611, wcs=wcs)
    ax1=plt.subplot(gs[:3,1:4])
    ax1.imshow(map12m,origin='low',cmap='coolwarm')
    #ax1.set_title('%s Map'%cmap)
    #ax1.add_compass(loc=4,color='black')
    ax1.axvline(x=x,color='r',ls='dashed',linewidth=3.0)
    ax1.annotate(x,(x+5,5),color='r',size=20)
    ax1.axhline(y=y,color='r',ls='dashed',linewidth=3.0)
    ax1.annotate(y,(5,y+5),color='r',size=20)
    #ax1.annotate(r'$T_{max}=$%0.2f'%(T[y,x]),(x+30,y+10),color='r',size='large')

    x_p=0.005

    ax2=plt.subplot(gs[3,:])
    y_p=s12.max()
    ax2.set_title(r'$^{12}CO$ // $FWHM=$%0.3f'%(FWHM12))
    ax2.plot(xx,s12,label='CO12')
    ax2.plot(xx[m],s12[m],'ko',label='Masked CO12 Data')
    ax2.plot(xx,gaussian(xx,popt12[0],popt12[1],popt12[2]),label='Fit to Masked Data')
    if (sd12<popt12).all():
        ax2.fill_between(xx,gaussian(xx,popt12[0]-sd12[0],popt12[1]-sd12[1],popt12[2]-sd12[2]),gaussian(xx,popt12[0]+sd12[0],popt12[1]+sd12[1],popt12[2]+sd12[2]),alpha=0.1)
    ax2.annotate(r'Fit Parameters for CO12: $A=$%0.3f +/-%0.3f, $x_0=$%0.3f +/-%0.3f, $\sigma=$%0.3f +/-%0.3f'%(popt12[0],sd12[0],popt12[1],sd12[1],popt12[2],sd12[2]),(x_p,y_p))
    ax2.axhline(y=popt12[0]/2.,xmin=popt12[1]-FWHM12/2.,xmax=popt12[1]+FWHM12/2.,color='r',label='HalfMaximum of fit CO12')
    #ax2.annotate(r'$FWHM12=$%0.3f'%(FWHM12),(x_p,1.5*y_p))
    #ax2.plot(np.linspace(0,1,zlen-2),der2,label='Second Derivative',alpha=0.5,ls='dashed')
    #ax2.plot(xx,np.where(m==False,popt12[0],0.0),label='Mask',alpha=0.3,ls='dashed')
    ax2.plot(xx,s13,alpha=0.7,label='CO13')
    ax2.plot(xx,s18,alpha=0.5,label='C18O')
    if (sd18<popt18).all():
        ax2.axvspan(popt18[1]-FWHM18/2.,popt18[1]+FWHM18/2.,alpha=0.25)
    ax2.legend()

    ax3=plt.subplot(gs[4,:])
    y_p=s13.max()
    ax3.set_title(r'$^{13}CO$ // $FWHM=$%0.3f'%(FWHM13))
    ax3.plot(xx,s13,'ko',label='CO13')
    ax3.plot(xx,gaussian(xx,popt13[0],popt13[1],popt13[2]),label='Fit to CO13')
    if (sd13<popt13).all():
        ax3.fill_between(xx,gaussian(xx,popt13[0]-sd13[0],popt13[1]-sd13[1],popt13[2]-sd13[2]),gaussian(xx,popt13[0]+sd13[0],popt13[1]+sd13[1],popt13[2]+sd13[2]),alpha=0.1)
    ax3.annotate(r'Fit Parameters for CO13: $A=$%0.3f +/-%0.3f, $x_0=$%0.3f +/-%0.3f, $\sigma=$%0.3f +/-%0.3f'%(popt13[0],sd13[0],popt13[1],sd13[1],popt13[2],sd13[2]),(x_p,y_p))
    ax3.axhline(y=popt13[0]/2,xmin=popt13[1]-FWHM13/2,xmax=popt13[1]+FWHM13/2,color='r',label='HalfMaximum of fit CO13')
    #ax3.annotate(r'$FWHM13=$%0.3f'%(FWHM13),(x_p,1.5*y_p))
    ax3.plot(xx,s18,alpha=0.5,label='C18O')
    if (sd18<popt18).all():
        ax3.axvspan(popt18[1]-FWHM18/2,popt18[1]+FWHM18/2,alpha=0.25)
    ax3.legend()

    ax4=plt.subplot(gs[5,:])
    y_p=s18.max()
    ax4.set_title(r'$C^{18}O$ // $FWHM=$%0.3f'%(FWHM18))
    ax4.plot(xx,s18,'ko',label='C18O')
    ax4.plot(xx,gaussian(xx,popt18[0],popt18[1],popt18[2]),label='Fit to CO18')
    if (sd18<popt18).all():
        ax4.fill_between(xx,gaussian(xx,popt18[0]-sd18[0],popt18[1]-sd18[1],popt18[2]-sd18[2]),gaussian(xx,popt18[0]+sd18[0],popt18[1]+sd18[1],popt18[2]+sd18[2]),alpha=0.1)
    ax4.annotate(r'Fit Parameters for CO18: $A=$%0.3f +/-%0.3f, $x_0=$%0.3f +/-%0.3f, $\sigma=$%0.3f +/-%0.3f'%(popt18[0],sd18[0],popt18[1],sd18[1],popt18[2],sd18[2]),(x_p,y_p))
    ax4.axhline(y=popt18[0]/2,xmin=popt18[1]-FWHM18/2,xmax=popt18[1]+FWHM18/2,color='r',label='HalfMaximum of fit CO18')
    #ax4.annotate(r'$FWHM18=$%0.3f'%(FWHM18),(x_p,1.5*y_p))
    if (sd18<popt18).all():
        ax4.axvspan(popt18[1]-FWHM18/2,popt18[1]+FWHM18/2,alpha=0.25)
    #ax4.plot(xx,s18f,label=r'Normalized $C^{18}O$ Spectra (Centered at $^{13}CO$ to cancel noise) ')
    ax4.legend()

    ax5=plt.subplot(gs[6,:])
    ax5.set_title(r'$^{12}CO$ Wings')
    #ax5.plot(xx[mask18],s12[mask18])
    ax5.fill_between(xx[xx<popt18[1]-FWHM18/2],s12[xx<popt18[1]-FWHM18/2],alpha=0.7)
    ax5.fill_between(xx[xx>popt18[1]+FWHM18/2],s12[xx>popt18[1]+FWHM18/2],alpha=0.7)

    plt.tight_layout()

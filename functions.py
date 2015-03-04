from astropy.io import fits
from astropy import wcs
from astropy.table import Table
import numpy as np
import pywcsgrid2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes,zoomed_inset_axes,mark_inset
from scipy.optimize import curve_fit
from IPython.display import clear_output, display, HTML

#===================Constants==================================================
ab12CO=77.        ##Abundancy ratio of 12co to 13co (Schoier et al. 2002 )
ab13CO=8.         ##Abundancy ratio of 13co to c18o (Schoier et al. 2002)
ab12CO_H2=8.5e-5     ##Abundancy ratio of 12CO to H2 (Frerking et al. 1982)
ab13CO_H2=1.25e-6    ##Abundancy ratio of 13CO to H2 (Moore et al. 2007)
ab18CO_H2=1.7e-7     ##Abundancy ratio of C18O to H2 (Frerking et al. 1982)
hk=0.0479924335     #* 10e9
h=6.626e-34         ##(J*s) Planck's const
k=1.381e-23         ##(J/K) Boltzman const
eo=8.85e-12         ##(F/m) permitivity const
Tbg=2.7             ##(K)   Background Temp
c=3.e8              ##(m/s) speed of light
Msol=1.98892e30     ##(kg)  Solar mass in kg
mH=1.67e-27         ##atomic hydrogen mass in kg
mH2=2*mH            ##(kg) mass of H2 in kg
mmw=2.33            ##mean mol weight wrt no of Hydrogen mols
SQARSCTSR=206265.**2 ##no of stradians in arcsec squared
dp=0.112*3.3365e30  ##(C*m) CO dipole moment (0.112 debye)
pac=3.08568e16      ##(m) one parsec in metres
v_co12_j32=345.796  #GHz
v_co13_j32=330.5879 #GHz
v_c18o_j32=329.331  #GHz
m_CO12=28.01 #atomic units
m_CO13=29.02 #atomic units
m_C18O=29.999 #atomic units
amu=1.66e-24 #g
k_b=1.38e-16 #erg/K

vres=0.83
velocity=np.array([-20.06933116, -20.90281358, -21.736296  , -22.56977842,
       -23.40326084, -24.23674326, -25.07022567, -25.90370809,
       -26.73719051, -27.57067293, -28.40415535, -29.23763777,
       -30.07112019, -30.90460261, -31.73808503, -32.57156745,
       -33.40504986, -34.23853228, -35.0720147 , -35.90549712,
       -36.73897954, -37.57246196, -38.40594438, -39.2394268 ,
       -40.07290922, -40.90639164, -41.73987405, -42.57335647,
       -43.40683889, -44.24032131, -45.07380373, -45.90728615,
       -46.74076857, -47.57425099, -48.40773341, -49.24121582,
       -50.07469824, -50.90818066, -51.74166308, -52.5751455 ,
       -53.40862792, -54.24211034, -55.07559276, -55.90907518,
       -56.7425576 , -57.57604001, -58.40952243, -59.24300485,
       -60.07648727, -60.90996969, -61.74345211, -62.57693453,
       -63.41041695, -64.24389937, -65.07738179, -65.9108642 ,
       -66.74434662, -67.57782904, -68.41131146, -69.24479388, -70.0782763 ])

#==============================================================================
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

def cube_to_max(cube,option=1):
    """
    Input: A 3D Masked Cube_Map
    Option 1: (X,Y)=(X-Pixels,Y-Pixels)
    Option 2: (X,Y)=(Y-Pixels,Velocity)
    Option 3: (X,Y)=(X-Pixels,Velocity)
    Return: Masked Maximum Intensity Map
    """
    xx=1 if option==2 else 2
    yy=1 if option==1 else 0
    X=cube.shape[xx]
    Y=cube.shape[yy]
    mapmax=np.ma.masked_array(np.zeros((Y,X)),mask=np.ones((Y,X)))
    for y in range(Y):
        for x in range(X):
            if (option==1):
                value = cube[:,y,x].max()
            elif  (option==2):
                value = cube[y,x,:].max()
            elif  (option==3):
                value = cube[y,:,x].max()
            else:
                print 'Options Only (1,2,3)'
                break
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

def save_to_fits(image,coords,filename):
    wheader=coords.to_header()
    hdu=fits.PrimaryHDU(data=image,header=wheader)
    hdu.writeto(filename)

def plot_maps(coords,fn,cm='coolwarm',norm='log',fs=(20,10),**kwargs):
    """
    Usage: plot_maps(maps=[list_of_maps],titles=[list_of_titles],fn=filename(put f for not saving),norm='log',fs=figsize,coords=wcs,cm=colormap)
    Input: 2D pixel Maps, WCS object of the Maps
    Output: Plot all Subplots of The Images with WCS Compass
    """
    plt.rcParams['figure.figsize'] = fs
    nm=len(kwargs['maps'])
    a=[[]]*nm
    for i,m in enumerate(kwargs['maps']):
        vmax=m.max()
        a[i]=(pywcsgrid2.subplot(1,nm,i,wcs=coords))
        a[i].grid()
        if norm=='log':
            im=a[i].imshow(m,origin='low',cmap=cm,norm=LogNorm(),aspect=1.)
        else:
            im=a[i].imshow(m,origin='low',cmap=cm,aspect=1.)
        a[i].set_title(kwargs['titles'][i])
        a[i].add_compass(loc=5,color='black')
        ains = inset_axes(a[i], width='2%', height='37%', loc=1)
        cb=plt.colorbar(im,cax=ains)
        plt.tight_layout()
        if (fn !='f'):
            plt.savefig(fn,bbox_inches='tight')

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
    Input: Frequency of observation in GHz, Pixel_map of Optical Thick Isotope, Cosmic Background Temperature
    Return: Pixel_map of Excitation Temperature
    """
    T0=hk*v
    A=Tr+T0/(np.exp(T0/Tbg)-1)
    Tx = T0/np.log(1+T0/A)
    return Tx


def Spectra(cube_map,y,x):
    """
    Spectrum of Selected Coordinates
    Input: Cube Map, pixel Coordinates
    Return: Vector of Spectrum Values
    """
    return cube_map[:,y,x]

def Spectra9(cube_map,y,x):
    """
    Mean Spectrum of 3x3 Selected Coordinates
    Input: Cube Map, pixel Coordinates
    Return: Mean Spectrum, Standard Deviation Spectrum
    """
    a=[]
    for j in [y-1,y,y+1]:
        for i in [x-1,x,x+1]:
            a.append(np.nan_to_num(cube_map[:,j,i]))
    a=np.array(a)
    return np.mean(a, axis=0),np.std(a, axis=0)

def MassEst(od,TR,Tex,d,res,j,i,X,fco):
    """
    od=optical depth
    TR=Integral
    Tex=Excitation Temp
    d=distance in kpc
    res=pixel resolution
    j,i=line
    X=CO to H2 abundance
    fco=frequency of line in Hz
    MG=MassEst(tau12_f,Integral12_fix,Tx12,d=2.,res=7.7,j=3,i=2,X=functions.ab12CO_H2,fco=functions.v_co12_j32*1e9)
    """
    dist=d*1e3*pac                         ##distance to the cloud in metres
    sq_pxsz=(res**2)*SQARSCTSR            ##squared pixel size in steradian
    ODcor=od/(1-np.exp(-od))               ##Optical depth correction
    """****************************************
     Set the Nij equation terms (result in m^-2)
    ****************************************"""
    term1=1.49*1e25                            ##
    term2=0.364*(Tex+0.922)                    ## the partition function Z
    term3=2.765*1e-9*(i*(i+1))                 ## h*vio/k
    term4=4.798*1e-11                          ## h/k
    Nco=(term1/(fco*j))*term2*np.exp(term3/Tex)*(1-np.exp(-term4*fco/Tex))*((1/(np.exp(term4*fco/Tex)-1))-(1/(np.exp(term4*fco/Tbg)-1)))**(-1)*ODcor*1e3*TR  ##should be in cm^-2
    MGas=(1E-4)*(1E-6)*2.8*mH2*dist*dist*sq_pxsz*X*Nco/Msol
    return MGas

def map_showXY(map12,map12m,map12my,map12mx,map13,map13m,map18,map18m,ta12,ta13,Tx12,Tx13,Tx18,wcs,y,x,dy,dx,dv,gf,s):
    """
    To use with IPython interact
    """

    #===================================Fitting======================
    #gf=0.2 #gooud fit parameter ~10%
    index_deviation=[10,2] #Max and Min Index Deviation for Second Derivative Mask
    xx=velocity
    #===========CO12=======================
    s12=Spectra(map12,y,x)
    #----mask------
    der2=np.diff(s12,2.) #Second Derivative
    m = np.ones(len(s12), dtype=bool)
    ind1=np.argsort(der2)[0]+1
    ind2=np.argsort(der2)[1]+1
    i=0
    while (np.abs(ind1-ind2)>index_deviation[0] or np.abs(ind1-ind2)<index_deviation[1]):
        #print np.abs(ind1-ind2),np.abs(ind1-ind2)>15
        ind2=np.argsort(der2)[i]+1
        i=i+1
    m[ind1:ind2]=False
    m[ind2:ind1]=False
    #---end of mask-----
    try:
        popt12, pcov12 = curve_fit(gaussian, xx[m], s12[m],p0=[s12.max(),xx[np.argmax(s12)],1.5],diag=(0.01,0.01))
    except:
        popt12,pcov12=np.zeros((3)),np.zeros((3,3))
    sd12= np.sqrt(np.diag(pcov12))      #Standard Deviation
    fit12 = (sd12<gf*np.abs(popt12)).all() #Good Fit?
    FWHM12=2.355*np.abs(popt12[2])      #Fitted Full Width Half Maximum
    FWTM12=1.865*FWHM12

    outflow=False
    if (FWTM12>4.):
	outflow=True

    FWHM12t=0.00001*2.355*np.sqrt(k_b*Tx12[y,x]/(m_CO12*amu)) #theoretical (thermal)
    ########################################
    # s912=Spectra9(map12,y,x)[0]
    # popt912, pcov912 = curve_fit(gaussian, xx, s912,p0=[s912.max(),xx[np.argmax(s12)],1.5],diag=(0.01,0.01))
    # sd912= np.sqrt(np.diag(pcov912))
    #===========CO13=======================
    s13=Spectra(map13,y,x)
    try:
        popt13, pcov13 = curve_fit(gaussian, xx, s13,p0=[0.25*popt12[0],popt12[1],popt12[2]],diag=(0.005,0.005))
    except:
        popt13,pcov13=np.zeros((3)),np.zeros((3,3))
    sd13= np.sqrt(np.diag(pcov13))      #Standard Deviation
    fit13 = (sd13<gf*np.abs(popt13)).all() #Good Fit?
    FWHM13=2.355*np.abs(popt13[2])      #Fitted Full Width Half Maximum
    FWHM13t=0.00001*2.355*np.sqrt(k_b*Tx13[y,x]/(m_CO13*amu)) #theoretical (thermal)
########################################################
    # s913=Spectra9(map13,y,x)[0]
    # popt913, pcov913 = curve_fit(gaussian, xx, s913,p0=[0.25*popt912[0],popt912[1],popt912[2]],diag=(0.1,0.1))
    # sd913= np.sqrt(np.diag(pcov913))
    #===========CO18=======================
    s18=Spectra(map18,y,x)
    try:
        popt18, pcov18 = curve_fit(gaussian, xx, s18,p0=[0.25*popt13[0],popt13[1],popt13[2]],diag=(0.001,0.001))
    except:
        popt18,pcov18=np.zeros((3)),np.zeros((3,3))
    sd18= np.sqrt(np.diag(pcov18))      #Standard Deviation
    fit18 = (sd18<gf*np.abs(popt18)).all() #Good Fit?
    FWHM18=2.355*np.abs(popt18[2])      #Fitted Full Width Half Maximum
    FWHM18t=0.00001*2.355*np.sqrt(k_b*Tx18[y,x]/(m_C18O*amu)) #theoretical (thermal)

    #================TABLE========================================
    col1=['$^{12}CO$ $T_{B}$','$^{13}CO$ $T_{B}$','$C^{18}O$ $T_{B}$',r'$\tau^{12}$',r'$\tau^{13}$','$^{12}CO$ $T_{X}$','$^{13}CO$ $T_{X}$',r'$^{12}CO$ FWHM',r'$^{12}CO$ FWTM',r'$^{13}CO$ FWHM',r'$C^{18}O$ FWHM',r'$^{12}CO$ Integral',r'$^{12}CO$ Core Integral', r'$^{12}CO$ Wings Integral','$^{12}CO$ Mass','$^{12}CO$ Core Mass','$^{12}CO$ Wings Mass']
    i12all=s12.sum()*vres
    i12=(s12[xx<popt13[1]-FWHM13/2].sum()+s12[xx>popt13[1]+FWHM13/2].sum())*vres if fit13 else np.nan
    i12core=i12all-i12 if fit13 else np.nan
    MG=MassEst(ta12[y,x],i12,Tx12[y,x],d=2.,res=7.7,j=3,i=2,X=ab12CO_H2,fco=v_co12_j32*1e9) if i12 else np.nan
    MGall=MassEst(ta12[y,x],i12all,Tx12[y,x],d=2.,res=7.7,j=3,i=2,X=ab12CO_H2,fco=v_co12_j32*1e9) if i12 else np.nan
    MGcore=MassEst(ta12[y,x],i12core,Tx12[y,x],d=2.,res=7.7,j=3,i=2,X=ab12CO_H2,fco=v_co12_j32*1e9) if i12 else np.nan

    if outflow:
        si='%0.2f $K\,km\,s^{-1}$'%i12
        sm='%0.2f $M_{\odot}$ // Ratio (all): %0.2f (core): %0.2f'%(MG,MG/MGall,MG/MGcore)

        siall='%0.2f $K\,km\,s^{-1}$'%i12all
        small='%0.2f $M_{\odot}$'%MGall

        sicore='%0.2f $K\,km\,s^{-1}$'%i12core
        smcore='%0.2f $M_{\odot}$'%MGcore
    else:
        si='0.0 (%0.2f) $K\,km\,s^{-1}$'%i12
        sm='0.0 (%0.2f) $M_{\odot}$'%MG

        siall='0.0 (%0.2f) $K\,km\,s^{-1}$'%i12all
        small='0.0 (%0.2f) $M_{\odot}$'%MGall

        sicore='0.0 (%0.2f) $K\,km\,s^{-1}$'%i12core
        smcore='0.0 (%0.2f) $M_{\odot}$'%MGcore

    col2=np.array(['%0.2f'%map12m[y,x],'%0.2f'%map13m[y,x],'%0.2f'%map18m[y,x],'%0.2f'%ta12[y,x],'%0.2f'%ta13[y,x],'%0.2f K'%Tx12[y,x],'%0.2f K'%Tx13[y,x],'%0.2f $km\,s^{-1}$ (thermal: %0.2f)'%(FWHM12,FWHM12t),'%0.2f $km\,s^{-1}$'%FWTM12,'%0.2f $km\,s^{-1}$ (thermal: %0.2f)'%(FWHM13,FWHM13t),'%0.2f $km\,s^{-1}$ (thermal: %0.2f)'%(FWHM18,FWHM18t),siall,sicore,si,small,smcore,sm])

    t=Table([col1,col2],names=('Name','Value'),meta={'name': 'first table'})
    display(t)
    #col1=np.array(['$^{12}CO$ $T_{B}$','%0.2f'%map12m[y,x]])
    #col2=np.array(['$^{13}CO$ $T_{B}$','%0.2f'%map13m[y,x]])
    #col3=np.array(['$C^{18}O$ $T_{B}$','%0.2f'%map18m[y,x]])
    #col4=np.array([r'$\tau^{12}$','%0.2f'%ta12[y,x]])
    #col5=np.array([r'$\tau^{13}$','%0.2f'%ta13[y,x]])
    #col6=np.array(['$^{12}CO$ $T_{X}$','%0.2f K'%Tx12[y,x]])
    #col7=np.array(['$^{13}CO$ $T_{X}$','%0.2f K'%Tx13[y,x]])
    #col8=np.array([r'$^{12}CO$ FWHM','%0.2f $km\,s^{-1}$ (thermal: %0.2f)'%(FWHM12,FWHM12t)])
    #col9=np.array([r'$^{13}CO$ FWHM','%0.2f $km\,s^{-1}$'%FWTM12])
    #col10=np.array(['$^{13}CO$ $T_{B}$','%0.2f $km\,s^{-1}$ (thermal: %0.2f)'%(FWHM13,FWHM13t)])
    #col11=np.array([r'$C^{18}O$ FWHM','%0.2f $km\,s^{-1}$ (thermal: %0.2f)'%(FWHM18,FWHM18t)])
    #col12=np.array([r'$^{12}CO$ Wings Integral',si])
    #col13=np.array(['$^{12}CO$ Wings Mass',sm])

    #t=Table([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13])
    #display(t)
    #t.write('test.tex',format='latex')

    #================Print==============================================
    # print 'tau12: %0.3f'%ta12[y,x]
    # print 'tau13: %0.3f'%ta13[y,x]
    # print '12CO Excitation Temperature: %0.2f'%Tx12[y,x]
    # print '13CO Excitation Temperature: %0.2f'%Tx13[y,x]
    # print 'C18O Excitation Temperature: %0.2f'%Tx18[y,x]
    # print '12CO FWHM: %0.2f km/s || Theoretical (thermal): %0.2f km/s (Ratio:%0.2f) || Larson L: %0.2f pc || Larson M: %0.2f Mo'%(FWHM12,FWHM12t,FWHM12/FWHM12t,(FWHM12/1.1)**(1/0.38),(FWHM12/0.42)**(1/0.2))
    # print '13CO FWHM: %0.2f km/s || Theoretical (thermal): %0.2f km/s (Ratio:%0.2f) || Larson L: %0.2f pc || Larson M: %0.2f Mo'%(FWHM13,FWHM13t,FWHM13/FWHM13t,(FWHM13/1.1)**(1/0.38),(FWHM13/0.42)**(1/0.2))
    # print 'C18O FWHM: %0.2f km/s || Theoretical (thermal): %0.2f km/s (Ratio:%0.2f) || Larson L: %0.2f pc || Larson M: %0.2f Mo'%(FWHM18,FWHM18t,FWHM18/FWHM18t,(FWHM18/1.1)**(1/0.38),(FWHM18/0.42)**(1/0.2))
    # if fit13:
    #     print 'Wings Integral in $^{12}CO$: %0.2f K'%(s12[xx<popt13[1]-FWHM13/2].sum()+s12[xx>popt13[1]+FWHM13/2].sum())

    #===========================================================================
    #===========PLOTS=======================
    #===========================================================================
    plt.rcParams['figure.figsize'] = 22, 35
    #dy,dx,dv=20,20,10
    vmax=np.argmax(s13)
    v1,v2=vmax-dv,vmax+dv

    #===========================================
    #===Map=====================================
    gs = gridspec.GridSpec(11, 8)

    #====ax1-Main Map===============================
    ax1=plt.subplot(gs[1:5,2:7])    #Axis for Main Map
    ax1.set_title('$^{12}CO$ Excitation Temperatures')
    #mim=ax1.imshow(map12m,origin='low',cmap='coolwarm')
    mim=ax1.imshow(Tx12.data,origin='low',cmap='coolwarm')       #Excitation Map
    ax1.axvline(x=x,color='k',ls='dashed',linewidth=1.0,alpha=0.6) #Current X
    ax1.annotate('$x$=%d'%x,(x+5,5),color='k',size=20)
    ax1.axhline(y=y,color='k',ls='dashed',linewidth=1.0,alpha=0.6)  #Current Y
    ax1.annotate('$y$=%d'%y,(5,y+5),color='k',size=20)
    ax1bar = inset_axes(ax1, width='2%', height='40%', loc=4)   #axis for colorbar
    ax1.annotate(s='', xy=(530,40), xytext=(579,40), arrowprops={'arrowstyle':'|-|','linewidth':2.0})
    ax1.text(542,43,'3 pc',color='white',fontsize=15)
    plt.colorbar(mim,cax=ax1bar)        #colorbar

    #=====================================
    axv0=plt.subplot(gs[0,2:7],sharex=ax1) #Axis for (X,Velocity) Map
    axv0.set_title('              $^{12}CO$ Max $T_{B}$')
    lev=np.linspace(0,map12mx.max(),20)     #Contourf Levels
    mimx=axv0.contourf(np.arange(0,map12mx.shape[1]),velocity,map12mx,levels=lev,cmap='coolwarm')
    axv0.xaxis.tick_top()
    axv0.axvline(x=x,color='k',ls='dashed',linewidth=1.0,alpha=0.75) #Current X

    axv1=plt.subplot(gs[1:5,7],sharey=ax1)  #Axis for (X,Velocity) Map
    axv1.set_title('$^{12}CO$ Max $T_{B}$')
    lev=np.linspace(0,map12my.max(),20)     #Contourf Levels
    mimx=axv1.contourf(velocity,np.arange(0,map12my.shape[1]),np.rot90(map12my,3),levels=lev,cmap='coolwarm')
    axv1.yaxis.tick_right()
    axv1.axhline(y=y,color='k',ls='dashed',linewidth=1.0,alpha=0.75) #Current Y

    #===========================
    #==Zoom=======
    x1, x2, y1, y2 = x-dx, x+dx, y-dy, y+dy     #Define Region
    region12=map12m[y1:y2,x1:x2]
    region13=map13m[y1:y2,x1:x2]
    if ((x+dx>350) and (y+dy>250)):     #dx=20, 7
        axins = zoomed_inset_axes(ax1, 140./np.max([dx,dy]), loc=3) #Axis for Zoom
    else:
        axins = zoomed_inset_axes(ax1, 140./np.max([dx,dy]), loc=1) #Axis for Zoom
    #axins.set_title('$^{12}CO$ and $^{13}CO$ (contours) \n max$T_{B}$ $(K)$')
    c12=axins.contourf(map12m,levels=np.linspace(region12.min(),region12.max(),15))
    c13=axins.contour(map13m,cmap='gnuplot',levels=np.linspace(region13.min(),region13.max(),7),alpha=0.75)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    mark_inset(ax1, axins, loc1=2, loc2=3, fc="none", ec="1.",lw=2.)

    axins.set_title('Max $T_{B}$ Contours:$^{13}CO$ \n Max $T_{B}$ Filled Contours:$^{12}CO$ ')
    # axins12 = inset_axes(axins, width='2%', height='25%', loc=2)
    # cbar12=plt.colorbar(c12,cax=axins12)
    # cbar12.ax.set_title('$^{12}CO$ \n maxT $(K)$')
    #
    # axins13 = inset_axes(axins, width='2%', height='25%', loc=1)
    # cbar13=plt.colorbar(c13,cax=axins13)
    # cbar13.ax.set_title('$^{13}CO$ \n maxT $(K)$')

    #==Map-Rectangles
    dd=0.5
    rect1 = [Rectangle((x-dx,y-dd), width=2.*dx, height=2.*dd, fill=False,color='red',linewidth=1.2,alpha=0.75)]
    axins.add_artist(rect1[0])
    rect2 = [Rectangle((x-dd,y-dy), width=2.*dd, height=2.*dy, fill=False,color='red',linewidth=1.2,alpha=0.75)]
    axins.add_artist(rect2[0])
    rect3= [Rectangle((x-1.5,y-1.5), width=3, height=3, fill=False,color='green',linewidth=1.5,alpha=0.95)]
    axins.add_artist(rect3[0])

    #=================3D Velocities==================================================
    #Parameters
    hl_lw=3 #Highlight LineWidth
    lev_18 = [popt18[0]/2.] #C18O Contour Display
    lev_13 = popt13[0]/2. #CO13 Contour Display
    min13=0.75
    min12=0.
    num_levels_12=10
    num_levels_13=8
    #======X-line=========================
    ax1y=plt.subplot(gs[5:7,2:])
    ax1y.set_ylabel('Velocity $(km\,s^{-1})$')
    ax1y.set_xlabel('X-Pixel')
    region13y=map13[v1:v2,y,x-dx:x+dx] #CO13 zoom region
    region12y=map12[v1:v2,y,x-dx:x+dx] #CO12 zoom region
    region18y=map18[v1:v2,y,x-dx:x+dx] #CO18 zoom region

    lev13y = np.linspace(np.abs(region13y.min())+min13,region13y.max(),num_levels_13)
    lev12y = np.linspace(min12,region12y.max(),num_levels_12)
    lev18y=lev_18

    cy=ax1y.contour(np.arange(x-dx,x+dx,1),velocity[v1:v2],region13y,levels=lev13y,linewidths=2.1,cmap='gnuplot',alpha=0.8)
    if fit13:
        ax1y.contour(np.arange(x-dx,x+dx,1),velocity[v1:v2],region13y,[lev_13],linestyles='--',linewidths=hl_lw,colors='red',alpha=1.)
    c2y=ax1y.contourf(np.arange(x-dx,x+dx,1),velocity[v1:v2],region12y,levels=lev12y)
    if fit18:
        ax1y.contour(np.arange(x-dx,x+dx,1),velocity[v1:v2],region18y,lev18y,linestyles='--',linewidths=hl_lw,colors='black',alpha=0.8)
        ax1y.contour(np.arange(x-dx,x+dx,1),velocity[v1:v2],region18y,[s18.max()*0.8],linewidths=hl_lw+1,colors='black',alpha=0.8)

    ax1y.plot(np.ones(velocity[v1:v2].shape)*x,velocity[v1:v2],'--')

    #===CO12 Colorbar Hacks======================================
    axinsy12 = inset_axes(ax1y, width='35%', height='3%', loc=2)
    cbary12=plt.colorbar(c2y,cax=axinsy12,orientation='horizontal')
    cbary12.ax.set_title('$^{12}CO$ $T_{B}$ $(K)$')

    #===CO13 Colorbar Hacks======================================
    axinsy13 = inset_axes(ax1y, width='30%', height='3%', loc=1)
    cbary13=plt.colorbar(cy,cax=axinsy13,orientation='horizontal')
    cbary13.ax.set_title('$^{13}CO$ $T_{B}$ $(K)$')

    #==========Y-line===============================================================
    ax1x=plt.subplot(gs[1:5,:2])
    ax1x.set_xlabel('Velocity $(km\,s^{-1})$')
    ax1x.set_ylabel('Y-Pixel')
    region13x=map13[v1:v2,y-dy:y+dy,x]
    region12x=map12[v1:v2,y-dy:y+dy,x]
    region18x=map18[v1:v2,y-dy:y+dy,x]

    lev13x = np.linspace(np.abs(region13x.min())+min13,region13x.max(),num_levels_13)
    lev12x = np.linspace(min12,region12x.max(),num_levels_12)
    lev18x=lev_18

    cx=ax1x.contour(velocity[v1:v2],np.arange(y-dy,y+dy),np.rot90(region13x,3),levels=lev13x,linewidths=2.1,cmap='gnuplot',alpha=0.8)
    if fit13:
        ax1x.contour(velocity[v1:v2],np.arange(y-dy,y+dy),np.rot90(region13x,3),[lev_13],linestyles='--',linewidths=hl_lw,colors='red',alpha=1.)
    c2x=ax1x.contourf(velocity[v1:v2],np.arange(y-dy,y+dy),np.rot90(region12x,3),levels=lev12x)
    if fit18:
        ax1x.contour(velocity[v1:v2],np.arange(y-dy,y+dy),np.rot90(region18x,3),lev18x,linestyles='--',linewidths=hl_lw,colors='black',alpha=0.8)
        ax1x.contour(velocity[v1:v2],np.arange(y-dy,y+dy),np.rot90(region18y,3),[s18.max()*0.8],linewidths=hl_lw+1,colors='black',alpha=0.8)
    ax1x.plot(velocity[v1:v2],np.ones(velocity[v1:v2].shape)*y,'--')

    #===CO12 Colorbar Hacks======================================
    axinsx12 = inset_axes(ax1x, width='3%', height='30%', loc=2)
    cbarx12=plt.colorbar(c2x,cax=axinsx12)
    cbarx12.ax.set_title('$^{12}CO$ \n  $T_{B}$ $(K)$')

    #===CO13 Colorbar Hacks======================================
    axinsx13 = inset_axes(ax1x, width='3%', height='30%', loc=1)
    cbarx13=plt.colorbar(cx,cax=axinsx13)
    cbarx13.ax.set_title('$^{13}CO$ \n $T_{B}$ $(K)$')
    axinsx13.yaxis.set_ticks_position("left")

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #============================================================================
    x_p=xx.min() #For Text Annotation

    ax2=plt.subplot(gs[7,:])
    y_p=s12.max()
    ax2.set_title('$^{12}CO$ // $FWHM=$%0.3f'%(FWHM12))
    ax2.plot(xx,s12,label='CO12')
    ax2.plot(xx[m],s12[m],'ko',label='Masked CO12 Data')
    ax2.plot(xx,gaussian(xx,popt12[0],popt12[1],popt12[2]),label='Fit to Masked Data')
    if fit12:
        ax2.fill_between(xx,gaussian(xx,popt12[0]-sd12[0],popt12[1]-sd12[1],popt12[2]-sd12[2]),gaussian(xx,popt12[0]+sd12[0],popt12[1]+sd12[1],popt12[2]+sd12[2]),alpha=0.25)
    ax2.annotate(r'Fit Parameters for CO12: $A=$%0.3f +/-%0.3f, $x_0=$%0.3f +/-%0.3f, $\sigma=$%0.3f +/-%0.3f'%(popt12[0],sd12[0],popt12[1],sd12[1],popt12[2],sd12[2]),(x_p,y_p))
    hline=np.linspace(popt12[1]-FWHM12/2,popt12[1]+FWHM12/2,10)
    ax2.plot(hline,np.ones(hline.shape)*popt12[0]/2,color='r',label='HalfMaximum of fit CO12')
    ax2.plot(xx,s13,alpha=0.7,label='CO13')
    ax2.plot(xx,s18,alpha=0.5,label='C18O')
    if fit18:
        ax2.axvspan(popt18[1]-FWHM18/2.,popt18[1]+FWHM18/2.,alpha=0.15)
    if fit13:
        ax2.axvspan(popt13[1]-FWHM13/2.,popt13[1]+FWHM13/2.,alpha=0.15,color='green')
    ax2.legend()

    ax3=plt.subplot(gs[8,:],sharex=ax2)
    y_p=s13.max()
    ax3.set_title('$^{13}CO$ // $FWHM=$%0.3f'%(FWHM13))
    ax3.plot(xx,s13,'ko',label='CO13')
    ax3.plot(xx,gaussian(xx,popt13[0],popt13[1],popt13[2]),label='Fit to CO13')
    if fit13:
        ax3.fill_between(xx,gaussian(xx,popt13[0]-sd13[0],popt13[1]-sd13[1],popt13[2]-sd13[2]),gaussian(xx,popt13[0]+sd13[0],popt13[1]+sd13[1],popt13[2]+sd13[2]),alpha=0.25)
    ax3.annotate(r'Fit Parameters for CO13: $A=$%0.3f +/-%0.3f, $x_0=$%0.3f +/-%0.3f, $\sigma=$%0.3f +/-%0.3f'%(popt13[0],sd13[0],popt13[1],sd13[1],popt13[2],sd13[2]),(x_p,y_p))
    hline=np.linspace(popt13[1]-FWHM13/2,popt13[1]+FWHM13/2,10)
    ax3.plot(hline,np.ones(hline.shape)*popt13[0]/2,color='r',label='HalfMaximum of fit CO13')
    ax3.plot(xx,s18,alpha=0.5,label='C18O')
    if fit18:
        ax3.axvspan(popt18[1]-FWHM18/2,popt18[1]+FWHM18/2,alpha=0.15)
    if fit13:
        ax3.axvspan(popt13[1]-FWHM13/2.,popt13[1]+FWHM13/2.,alpha=0.15,color='green')
    ax3.legend()

    ax4=plt.subplot(gs[9,:],sharex=ax2)
    y_p=s18.max()
    ax4.set_title(r'$C^{18}O$ // $FWHM=$%0.3f'%(FWHM18))
    ax4.plot(xx,s18,'ko',label='C18O')
    if fit18:
        ax4.plot(xx,gaussian(xx,popt18[0],popt18[1],popt18[2]),label='Fit to CO18')
        ax4.fill_between(xx,gaussian(xx,popt18[0]-sd18[0],popt18[1]-sd18[1],popt18[2]-sd18[2]),gaussian(xx,popt18[0]+sd18[0],popt18[1]+sd18[1],popt18[2]+sd18[2]),alpha=0.25)
        ax4.annotate(r'Fit Parameters for CO18: $A=$%0.3f +/-%0.3f, $x_0=$%0.3f +/-%0.3f, $\sigma=$%0.3f +/-%0.3f'%(popt18[0],sd18[0],popt18[1],sd18[1],popt18[2],sd18[2]),(x_p,y_p))
        hline=np.linspace(popt18[1]-FWHM18/2,popt18[1]+FWHM18/2,10)
        ax4.plot(hline,np.ones(hline.shape)*popt18[0]/2,color='r',label='HalfMaximum of fit CO18')
        ax4.axvspan(popt18[1]-FWHM18/2,popt18[1]+FWHM18/2,alpha=0.15)
    ax4.legend()

    ax5=plt.subplot(gs[10,:],sharex=ax2)
    ax5.set_title(r'$^{12}CO$ Wings')
    ax5.fill_between(xx[xx<popt13[1]-FWHM13/2],s12[xx<popt13[1]-FWHM13/2],alpha=0.7)
    ax5.fill_between(xx[xx>popt13[1]+FWHM13/2],s12[xx>popt13[1]+FWHM13/2],alpha=0.7)
    if outflow:
	ax5.annotate(r'Outflow Mass Estimation: %0.2f $M_{\odot}$'%MG,(x_p,y_p) ,fontsize=17)

    #ax6=plt.subplot(gs[11:,:],sharex=ax2)
    #ax6.set_title('3X3 Mean and Standard Deviation Spectrum')
    #ave12=Spectra9(map12,y,x)
    #ave13=Spectra9(map13,y,x)
    #ave18=Spectra9(map18,y,x)
    #ax6.plot(xx,ave12[0],color='blue',label='CO12 Mean')
    #ax6.fill_between(xx,ave12[0]-ave12[1],ave12[0]+ave12[1],color='blue',alpha=0.25)
    #ax6.plot(xx,ave13[0],color='green',label='CO12 Mean')
    #ax6.fill_between(xx,ave13[0]-ave13[1],ave13[0]+ave13[1],color='green',alpha=0.25)
    #ax6.plot(xx,ave18[0],color='red',label='CO12 Mean')
    #ax6.fill_between(xx,ave18[0]-ave18[1],ave18[0]+ave18[1],color='red',alpha=0.25)
    # ax6.plot(xx,gaussian(xx,popt913[0],popt913[1],popt913[2]),'--',linewidth=3.,label='Fit')
    #ax6.legend()

    plt.tight_layout()
    if s:
        plt.savefig('full%d-%d'%(y,x),bbox_inches='tight')
        #t.write('t%d-%d.tex'%(y,x),format='latex')

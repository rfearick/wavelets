from __future__ import print_function
from __future__ import division

#import Wavelets
from Wavelets import *
import numpy as np
import matplotlib as mpl
import pylab as plt
from scipy.stats import chi2
from matplotlib.colors import LightSource


"""
A Spectrum class for wavelet analysis of nuclear data

(14/12/2017)
remove parts no longer used
(26/6/2017)
From uct2016b/wavelet

(20/10/2016)
From uct2015a/newplots

(26/6/2014)
From uct2014a/develop/SpectrumAnalyserPbCleanup.py
Restore factor of 3, rather than 208pb hack in wavelet/fourier scale.
Peak fit using polyfit

(28/11/2013)
Cleanup after hack for 208Pb gdr check against I.P's thesis.

(27/6/2013)
add placeholder for spectrum limits if inside ROI
(21/6/2013)
continue clean up
(27/5/2013)
SpectrumAnalyser3.py
Clean up prior to Da trip.
(15/5/2013)
Use rescaled power plots (divide power by scale).
(7/5/2013) SpectrumAnalyser2
from fig2.py
(6/5/2013)
update colours and add contours.
improve power 2-d plot in triple plot
handle more and less scales
(2/5/2013)
update pwr section in triple plot
(3/4/2013) fig2.py
from SpectrumAnalyser.py for paper.
(2/4/2013)
restore original triple style for paper.
(23/1/2013)
Now from Spectrum6
(25/1): change some arguments around.
        add option to omit peak shoulders in search

Original from:
thisfilename="rcnp-full-8a.py"

(23/1/2013)
Spectrum6
More restructuring
(22/1/2013)
Spectrum5
Improved output,
cleanup of code.
Note bug: N=2401 trncates a at 2400, but it is later overlayed by A
must fix this ...

(21/1/2013)
Remove interactive features that appeared in uct2012b/Spectrum3.py

(10/12/12)
use AxesGrid for standard 3-pane plot.
(30/11/2012)
Include changes for rcnp-full-9

"""

"""
Spectrum class
(29/11/2012) at last.

TO FIX:

Nhi-Nlo must be even?

(postamble)(done)
now
thisfilename

"""


# ------------------------------------------------------

class Spectrum(object):
    """
    Spectrum class for wavelet analysis
    
    A place holder for analysis data
    """
    
    def __init__(self,wavelet,data,Nlo,Nhi,sigma,ndec,smooth,
                 order=2,scaling="log", scaletype="fourier",scalepwr=None):
        """
        Initialize a Spectrum object.

        Parameters
        ----------
        wavelet : wavelet object
                  Wavelet object used for transform. Usually Morlet
        data : ndarray of floats, or data file name,
                  data to be transformed
        Nlo, Nhi : int
                  Low and high indices giving region of ibnterest in `data`
        sigma : float
                  Standard deviation of spectral features in `data`.
                  Used in construction of model reference spectra
        ndec : int
                  Decimation value --- compress `data` by summing over
                  `ndec` elements.
        smooth : float
                  Smoothing parameter for wavelet smoothing (deprecated/no longer used)
        order : int
                  Order of `wavelet` when family has variable order (e.g. paul, DOG) 
        scaling : str
                  'linear' or 'log' scaling of wavelet scale.
        scaletype : str
                  'fourier' or 'wavelet' scale used for wavelet transform.
        scalepwr : bool
                   True if power spectrum normalised by scale.

        Notes
        -----

        Utility object used in wavelet analysis of nuclear spectra.

        References
        ----------
        """
        if smooth>0.0:
            raise(ValueError("smooth not in use"))

        # basic attributes (arg list and defaults)
        self.Morlet=Morlet  # wavelet types (only one for now)
        self.sigma=sigma
        self.scalepwr=scalepwr
        self.datafilename=data

        # This is set up for our spectra and needs generalising!
        Ns=3000  # default: 30.0 MeV at 10 keV/ch. May need tuning. 
        if Nhi>Ns:
            raise(ValueError("Nhi<Nlo"))
        dN=Nhi-Nlo
        if (dN//2)*2 != dN:
            raise(ValueError("Nhi-Nlo not even"))

        # Read in the data
        self.dataLo=0
        self.dataHi=0
        if isinstance(data,str):
            self.datafilename=data
            if ndec!=0: # decimate data
                (A,chperE)=self.Open(data,Ns*ndec,ndec)
            else:
                (A,chperE)=self.Open(data,Ns)
        elif isinstance(data,np.ndarray):
            self.datafilename=None
            A=data
            chperE=100.0
        else:
            raise(ValueError("file name?"))
            
        dN=Nhi-Nlo
        if (dN//2)*2 != dN:
            Nhi+=1

        # Keep a copy of the raw data
        self.basicdata=A.copy()
        self.chperE=chperE

        # Now do data preprocessing prior to wavelet transform

        # some spectra have a restricted range. continue the lower and
        # upper values to the ends. then reflect.
        # first eliminate upper cutoff in data by replacing with const
        # (fix for rcnp spectra)
        meanlo=np.sum(A[Nlo:Nlo+10])/10.0
        meanhi=np.sum(A[Nhi-9:Nhi+1])/10.0
        ## for i in range(Nlo+1):
        ##     x=float(-(Nlo-i))/100.0
        ##     A[i]=meanlo
        for i in range(Nhi,Ns):
             x=float(-(i-Nhi))/100.0
             A[i]=meanhi
        # Truncate data to internal maximum size
        A=A[0:Ns]
        #now reflect data into upper half of array a0 for better periodicity
        a0=np.zeros(2*Ns)
        a0[0:Ns]=A[0:Ns]
        a0[Ns:]=A[::-1]
        # subtract smoothed version of spectrum if requested
        self.a=a0

        # Now focus on Region Of Interest in the data
         
        # Statistical features of data in ROI
        self.meana=np.mean(self.a[Nlo:Nhi])
        self.vara=np.var(self.a[Nlo:Nhi])
        
        # Straight FFT of ROI 
        Nr=Nhi-Nlo
        tempft=self.a[Nlo:Nhi]-np.sum(self.a[Nlo:Nhi])/(Nhi-Nlo)
        norm1=np.sum(tempft*tempft)
        self.datafft=np.fft.fft(tempft)
        norm2=np.sum(self.datafft*self.datafft.conjugate()).real/len(self.datafft)
        print('power(t)',norm1)
        print('power(f)',norm2)
        
        # Wavelet transform the data -- CWT all and then keep ROI
        maxscale=4  # !!!!!!!!!! abitrary value tuned for our data
        wscaling='log'
        if scaling=='linear' or scaling=='linlin': wscaling='linear'
        notes=0
        if wscaling=='log': notes=16  # chosen as a reasonable value
        # keep wavelet object
        usederiv=0
        self.cw=wavelet(self.a,largestscale=maxscale,notes=notes,order=order,
                        scaling=wscaling, deriv=(usederiv==1))
        # keep wavelet's fft of data in ROI
        self.datahat=self.cw.fftdata
        # Only retain data in ROI
        self.a=A[Nlo:Nhi]

        # calculate entropy index
        self.entindex=self.EntropyIndex(self.cw,Nlo,Nhi)  
        # keep local reference to scales  (in channels)
        self.scales=self.cw.getscales()
        self.datapwr=(self.datahat*self.datahat.conjugate()).real
        # keep local ref. to wavelet coefficients in ROI
        self.cwt=self.cw.getdata()
        self.cwt=self.cwt[:,Nlo:Nhi]
        # leep local reference to wavelet power in ROI
        self.pwr=self.cw.getpower()
        self.pwr=self.pwr[:,Nlo:Nhi]
        # calculate wavelet scale power spectrum from ROI
        self.scalespec=np.sum(self.pwr,1)/(Nhi-Nlo)
        # calculate variance on each scale in ROI
        self.variances=np.zeros(len(self.scales))
        for i in range(len(self.scales)):
            self.variances[i]=np.var(self.cwt.real[i,:]/np.sqrt(self.scales[i]))
        #
        # where does normfactor come from (1/7/11) (T&C)?
        normfactor=(1.0/16.0)/0.776/np.pi**(-0.25)
        print("normfactor",normfactor)
        norm4=np.sum(self.scalespec)
        print("norm4", norm4)
        norm5=np.sum(self.scalespec/self.scales)*(Nhi-Nlo)*normfactor
        print("norm5", norm5)
        variances=normfactor*np.sum(self.pwr,0)
        norm3=np.sum(variances)
        print('norm3',norm3)
        self.variances = variances*norm1/norm3
        print(sum(self.variances))
        print('variance',self.vara,' mean',self.meana)

        # calculate energy and scales for axes in ROI
        self.x=np.arange(float(Nlo),float(Nhi+1))/chperE
        self.y=self.cw.fourierwl*self.scales/chperE
        if scaletype=="wavelet":
            print("selecting wavelet scale")
            self.y=self.y/3.0  ####/2.0/0.8 ####################  8/11 ???
        # scale power spectrum by wavelet scale if needed
        if scalepwr == 'scale':
            print("scaling scalespec")
            self.scalespec /= self.y
        # calculate the fft power spectrum in the ROI on a wavelet scale (!)
        sfft=self.datafft  # fft of data in roi
        fftlen=len(sfft)
        Nf=fftlen       #1/7/11
        Nfplot=Nf//2
        # ************  0 freq in below?????????
        #fftscales=1.0/np.fft.fftfreq(Nf)
        xfft0=np.arange(1.0,float(Nf//2),1.0)*self.chperE/Nf
        if scaletype=="wavelet":
            xfft0=xfft0*3.0
        self.xfft=1.0/xfft0  #/3.0
        pwrdata=(sfft*sfft.conjugate()).real/Nf
        pwrdata[1:Nf//2]=(pwrdata[1:Nf//2]+pwrdata[Nf-2:Nf//2-1:-1])/2.0
        self.pwrdata=pwrdata[1:Nf//2]
        
        # following added 13/7/2012: estimate variance of stat. noise
        # checked: self.vara=v1+v2
        Gh=self.a
        Gsh=self.convolute(Gh,self.gauss,2.0)
        Gd=Gsh-Gh
        v1=np.var(Gsh)
        # v2 is roughly the variance due to statistical noise
        #at some stage should use dwt and get this from D[0]
        v2=np.var(Gd)        
        print("stats",self.meana,self.vara, v1, v2, v1+v2)
        self.varnoise=v2
        self.varsignal=self.vara-v2

        # confidence limits
        noise=self.varnoise/self.vara
        Nci=len(self.y)
        # take into account gaussian fold
        if scaletype=="wavelet":
            Pk=self.referencespectrum(Nci, sigma/0.01)
            spfilter=self.spectrumfilter( Nci, self.y, sigma/3.0 )
        else:
            Pk=self.referencespectrum(Nci, sigma/0.010)
            spfilter=self.spectrumfilter( Nci, self.y, sigma )
        ci=self.confidencelimit( Pk, Nhi-Nlo, self.y )

        #13/7/12: new cipn has calculated noise contribution
        self.cipn=(self.varsignal/self.vara)*ci*spfilter**2+(self.varnoise)/self.vara
        self.Pkpn=0.5*Pk*((self.varsignal/self.vara)*spfilter*spfilter+self.varnoise/self.vara)

        
    def Open( self, name, points, ndec=0 ):
        """
        open spectrum file and optionally decimate.

        Parameters
        ----------
        name : str
               Name of file from which data is read.
        points : int
               (Sometimes) minimum length of returned data array
        ndec : int
               Decimation value --- compress `data` by summing over
               `ndec` elements if greater than zero.

        Returns
        -------
        newdata : ndarray of floats
                Data read from file, equally spaced in energy.
        chpermev : float
                Data channels (elements) per MeV.
        
        Notes
        -----
        returns 3000 channel spectrum by default.
        Data origin guessed from name. Highly application specific.
        (28/06/2012) Fix cross section change when compressed.
        """
        chpermev=100.0
        if name == "TEST1":
            N=3000
            M=2
            Gd2=0.5 # Gamma/2
            E0=20.0
            energy=np.arange(0.0,30.0,0.01)
            newdata=Gd2**2/((energy-E0)**2+Gd2**2)
        elif name=="TEST2":
            N=3000
            M=2
            T=3.0/10.0
            energy=np.arange(0.0,30.0,0.01)
            newdata=np.sin(2.0*np.pi*energy/T)+0.5
        elif name=="TEST3":
            N=3000
            M=2
            T=30.0/10.0
            energy=np.arange(0.0,30.0,0.01)
            newdata=5.0*np.ones(len(energy))
            Nsq=80
            for i in range(0,len(energy),Nsq*2):
                newdata[i:i+Nsq]=0.0
        elif "5keV" in name:
            energy, data, uncertainty=np.loadtxt(name, unpack=True)
            N=len(data)
            M=2
            New=int(N//M)
            newdata=np.zeros(New)
            i=0
            for j in range(New):
                sum=0.0
                for k in range(M):
                    sum+=data[i]
                    i+=1
                newdata[j]=sum/M
            chpermev=200.0/M
        elif "_IVD_" in name or "BE1" in name:
            try:
                energy, BE1, K=np.loadtxt(name,unpack=True)
            except:
                energy, BE1=np.loadtxt(name,unpack=True)
            print( "len energy", len(energy))
            tot=0.0
            eo=0.0
            E=[]
            B=[]
            for i,e in enumerate(energy):
                if np.fabs(e-eo)<0.00001:
                    tot += BE1[i]
                else:
                    if eo!=0:
                        E.append(eo)
                        B.append(tot)
                    tot=BE1[i]
                eo=e
            self.EBE1=np.array(E)
            self.BE1=np.array(B)
            #energy=np.array(E)
            #BE1=np.array(B)
            chpermev=100.0  # 10 keV/ch
            origdata=np.zeros(5000)  # 50 MeV of data
            print("d")
            for e,b in zip(E,B):
                i=int(e*chpermev+0.001)
                if i>=5000: break
                origdata[i]+=b
           # kludges from globals
            print(self.sigma)
            fwhm=(self.sigma/1000.0)*chpermev
            print("e")
            newdata=self.convolute(origdata, self.gauss,fwhm/2.35)
            print("f")           
        else:
            #energy, data, uncertainty=np.loadtxt(name, unpack=True)
            NN=6000
            
            energy=np.linspace(0.0,29.99,NN)
            data=np.zeros(NN)
            try:
                e, d=np.loadtxt(name, unpack=True)
            except:
                e, d, deld=np.loadtxt(name, unpack=True)
            e+=0.0025  # 31/10/13
            Ne0=int(e[0]*200+0.01)
            print("start",Ne0,e[0])
            self.dataLo=Ne0
            self.dataHi=Ne0+len(e)
            energy[Ne0:Ne0+len(e)]=e
            data[Ne0:Ne0+len(e)]=d
            data[Ne0+len(e):]=d[-1]
            data[:Ne0]=d[0]
            #data=data-39.0
            #data=data-79.0
            data=data-np.amin(data)+1.0
            N=len(data)
            print(N)
            M=2
            New=int(N//M)
            newdata=np.zeros(max(New,points))
            i=0
            for j in range(New):
                sum=0.0
                for k in range(M):
                    sum+=data[i]
                    i+=1
                newdata[j]=sum#/M
            chpermev=200.0/M
            print( chpermev,1.0/(energy[2]-energy[1]))
        print("input",len(newdata), chpermev)
        return newdata,chpermev


    def gauss( self, h, sig ):
        """
        Return Gaussian array with standard deviation sig in form useful for fft

        Parameters
        ----------
        h : ndarray of floats
             Array to be filled with function in fft ordering.
        sig :
             Standard deviation of Gaussian.
        
        Notes
        -----
        10/7/2012: fixed according to [da2011]/testconvolution.py
        """
        width = 8      # width*sig is range where gaussian is "nonzero"
        fact=1.0/(np.sqrt(2.0*np.pi)*sig)
        M=int(width*sig)
        N=len(h)
        h[0]=fact
        for i in range(1,M):
            x=float(i)
            h[i]=fact*np.exp(-x*x/(2.0*sig*sig))
            h[N-i]=h[i]


    def lorentzian( self, h, sig ):
        """
        Return lorentzian (breit-wigner) with width sig in
        form useful for fft

        Parameters
        ----------
        h : ndarray of floats
             Array to be filled with function in fft ordering.
        sig :
             Standard deviation of Gaussian. Converted to FWHM using
             Gaussian approximation.
        
        Notes
        -----
        10/7/2012: fixed according to [da2011]/testconvolution.py
        """
        width = 15      # width*sig is range where lorentzian is "nonzero"
        sig=sig*2.3/2.0
        fact=sig/np.pi
        M=int(width*sig)
        N=len(h)
        h[0]=fact/sig/sig
        for i in range(1,M):
            x=float(i)
            h[i]=fact/(x*x+sig*sig)
            h[N-i]=h[i]


    def EntropyIndex(self,cw,Nlo,Nhi):
        """
        Returns the entropy index calculated from the
        wavelet power spectrum
        At channels Nlo:Nhi
        """
        Nc=Nhi-Nlo
        coeff=np.sqrt(cw.getpower())
        coeff=coeff[Nlo:Nhi,:]
        meancoeff=np.sum(abs(coeff),0)/Nc
        #print meancoeff
        fluctcoeff=coeff#/meancoeff[NewAxis,:]
        maxindex=float(Nc)*np.log(float(Nc))/Nc
        return np.sum(fluctcoeff*np.log(fluctcoeff),0)/(Nc*maxindex)

    def Smooth( self, spectrum, sigma ):
        """
        Returns a spectrum smoothed by convolution with a gaussian of
        width sigma
        """
        h=np.zeros(len(spectrum))
        # fill h with convoluting gausian
        self.gauss(h,sigma)
        # convolute
        fh=np.fft.fft(h)
        fg=np.fft.fft(spectrum)
        return np.fft.ifft(fh*fg).real

    def convolute(self, spectrum, lineshape, width):
        """
        convolute spectrum with lineshape of s.dev. width
        note that width is in channels
        """
        Nspect=len(spectrum)
        g=np.zeros(Nspect*2)  
        h=np.zeros(Nspect*2)    
        g[0:Nspect]=spectrum[0:Nspect]
        lineshape(h,width)
        fh=np.fft.fft(h)
        fg=np.fft.fft(g)
        G=abs(np.fft.ifft(fh*fg))      # convoluted spectrum
        G=G[0:Nspect]               # truncate to size
        return G
    
    def autocorrelation(self, Gw):
        """
        calculate autocorrelation.
        """
        fg=np.fft.fft(Gw-np.mean(Gw))
        Nfg=len(Gw)
        meand=np.mean(Gw)
        Accomplex=(np.fft.ifft(fg*np.conjugate(fg))) / float(Nfg) #- 1.0
        Ac=Accomplex.real/meand**2
        return Ac-1.0
    
    # ----------------------- spectrum -------------
    def referencespectrumrednoise(self, nci, alpha=0.7 ):
        yc=np.pi*np.array(np.arange(nci,0,-1)-1,'float')/float(nci)
        Pk=(1.0-alpha**2)/(1.0+alpha**2-2.0*alpha*np.cos(yc))
        Pk[:]=(1.0-alpha**2)/(1.0-alpha)**2
        return Pk

    # !!!!!!!!!!!!!!!!!**********************************************
    def referencespectrum(self, nci, sig):
        """
        27/7/2010: two sources of variance:
            1) fluctuations due to PT
            2) counting noise
            both with mean=var
            hence basic Pk=2
        29/6/2012: taken from spectrum-analysi-8.py (da2012)
        30/11/2012: restored to noiseless form (line 3) from line 1.
        """
        Pk=np.ones(nci)*1.0/(1.0/(2.0*np.sqrt(np.pi)*sig)+0.5)
        ####Pk=np.ones(nci)*1.0/(1.0/(2.0*np.sqrt(np.pi)*sig)+1)
        #####Pk=np.ones(nci)*2.0*np.sqrt(np.pi)*sig
        return Pk

    def confidencelimit(self, refspec, Ntimes, scales ):
        """
        Calculate CI on power spectrum
        Ntimes: no. of times over which data is averaged (Nhi-Nlo)
        scales: 
        refspec: mean reference spectrum
        """
        gamma=2.32 # decorrelation length - T&C for Morlet
        p=0.05 # 95% ci
        nu=2.0*np.sqrt(1.0+(float(Ntimes)/gamma/scales)**2)
        df=nu
        rv=chi2(df).ppf  # percentage point function - use to get 95% upper cl
        chisq=rv(1.0-p/2)
        ci=refspec*(chisq/nu)
        return ci

    def spectrumfilter(self, ndata, scales, sig ):
        """
        provide a cutoff for high freq (small scales) in spectra, due to
        convolution with resolution function
        """
        sfilter=np.exp(-4.0*np.pi**2*sig**2/(scales*scales)/2)
        return sfilter

class PeakFinder(object):
    def __init__(self, spectrum, y, ylimit, variance, cilimit, doshoulders):
        # locate peaks with crude algorithm
        self.peaks=[]
        self.peaksig=[]
        self.peakamplitude=[]
        dd1=spectrum[2]-spectrum[0]
        dd2=spectrum[2]-2.0*spectrum[1]+spectrum[0]
        #ddscale=S.y[1:-1]
        ddscale=spectrum
        self.deriv1=np.zeros(len(ddscale))
        self.deriv1[0]=dd1
        self.deriv2=np.zeros(len(ddscale))
        self.deriv2[0]=dd2
        
        for i in range(1,len(y)-2):
            d1= spectrum[i+2]-spectrum[i]
            d2= spectrum[i+2]-2.0*spectrum[i+1]+spectrum[i]
            if  d1*dd1<0.0 and (d2<0.0 or dd2<0):
                pq=np.polyfit(y[i-1:i+3],spectrum[i-1:i+3],2)
                ymax=-pq[1]/2.0/pq[0]
                print(ylimit, ymax, cilimit[i+1], spectrum[i+1]/variance)
                if y[i+1]>ylimit:
                    self.peaksig.append(spectrum[i+1]/variance>cilimit[i+1])
                    self.peaks.append(ymax)
                    self.peakamplitude.append(spectrum[i+1])
            if doshoulders:
                # not entirely clear where we should place peaks below: note choice!
                if d2*dd2<0.0 and dd2<0.0 and d1>0.0 and dd1>0.0:  # put at i
                    print("....", y[i], cilimit[i], spectrum[i]/variance)
                    if y[i]>ylimit:
                        self.peaksig.append(spectrum[i]/variance>cilimit[i])
                        self.peaks.append(y[i])
                        self.peakamplitude.append(spectrum[i])
                if d2*dd2<0.0 and d2<0.0 and d1<0.0 and dd1<0.0:   # put at i+2
                    print("....", y[i+2], cilimit[i+2], spectrum[i+2]/variance)
                    if y[i+2]>ylimit:
                        self.peaksig.append(spectrum[i+2]/variance>cilimit[i+2])
                        self.peaks.append(y[i+2])
                        self.peakamplitude.append(spectrum[i+2])
                
            dd1=d1
            dd2=d2
            self.deriv1[i]=d1
            self.deriv2[i]=d2
        print("Peaks found, ", len(self.peaks))

if __name__=="__main__":
    print("Version of 26/6/2017")
    import pylab as plt
    from Plotter import *

    # run control (bool) and user settings
    printer=0         # print plots to a file
    plotcoeff=0       # plot coefficients or power
    plotsmooth=0      # show smoothed spectrum over spectrum
    plotentropy=0     # plot entropy index or scale spectrum
    usesmooth=0       # use spectrum-smoothed
    plotspectrum=1    # plot spectrum or autocorrelation
    normtovar=1       # normalize power spectrum to variance
    usederiv=0
    waveletclass=Morlet
    #waveletclass=GaussPW
    order=2
    fitpeaks=True
    plotci=False
    outputscalespectrum=True
    usewaveletscale=True
    useLatex=True
    #usewaveletscale=False
    doshoulders=False



    datafile="../../uct2011b/data/5keV/2800.dat"
    datafile="TEST2"
    sig=0.03/2.35 # 50 keV
    sigsm=2.5/2.35    # width of smoothing gaussian (MeV)
    Nf=512            # channels in FFT for power spectrum
    maxscale=4        # maximum scale is Ns/maxscale
    notes=16          # notes per octave. 0 gives linear scale

    # region of interest is gdr
    Nlo=1000
    Nhi=2024
    scaling='log'

    set_plot_text_method(useLatex)

    # read in data and do wavelet transform
    sigsm=0.0
    if usewaveletscale:
        scaletype="wavelet"
    else:
        scaletype="fourier"
    Nlo=1600
    Nhi=2400
    Mlo=1500
    Mhi=2500
    #datafile="TEST1"
    S=Spectrum(waveletclass, datafile, Nlo, Nhi, sig, 0, sigsm,order=order,scaletype=scaletype,scalepwr='scale')
    print("data",S.a[698:702])
    print(S.scales[0:5])
    # extract data
##            avspectrum=np.sum(S.a)/len(S.a)
##            normfactor=(1.0/16.0)/0.776/np.pi**(-0.25)
##            variances=normfactor*np.sum(S.pwr,0)/(Nhi-Nlo)
##            norm3=np.sum(variances)

    siglimit=sig*2.35*2  # lowest energy for worthwhile peaks
    if fitpeaks:
        Peaks=PeakFinder(S.scalespec, S.y, siglimit, S.vara, S.cipn, doshoulders)
    else: Peaks=None

    # Plots ------------------------

    set_plot_text_method(useLatex)
    plot_triple(S, Peaks=Peaks, plotlim=(Mlo,Mhi),plotcoeff=plotcoeff, plotci=plotci,
                plotentropy=plotentropy, normtovar=normtovar,
                useLatex=useLatex, usewaveletscale=usewaveletscale, scaling="log",
                title=r"!\LARGE$^{28}$Si")
    plot_scale_spectrum(S, Peaks=Peaks, normtovar=normtovar, plotci=plotci,
                        usewaveletscale=usewaveletscale)
    plt.show()
  

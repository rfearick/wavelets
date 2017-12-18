from __future__ import print_function
from __future__ import division

from Wavelets import *
import numpy as np
import matplotlib as mpl
import pylab as plt
from scipy.stats import chi2
from matplotlib.colors import LightSource

# taken from uct2016b/wavelet/Plotter.py 26/6/2017
# taken from uct2015a/newplots/Plotter.py 20/10/2016
# taken from uct2014a/develop/Plotter.py 2/6/2014
# taken from uct2913c/pb208/fig4.py, 29/5/2014.


# =====================================================================
def plot_triple(S, Peaks=None, plotlim=(800,1900),
                plotcoeff=True, plotci=False, plotentropy=False,
                normtovar=True, useLatex=False, usewaveletscale=True, scaling=None,
                title=None, outfilename=None,maxspect=None,limits_scale=None,
                clip_fraction=1.0, ylabels=False):
    import pylab as plt
    import matplotlib.lines as lines
    import datetime
    import inspect

    """
    Wavelet analysis "triple plot": Three plots
       [ Title area    ]     [  Spectrum              ]
       [ Wavelet power ]     [  Wavelet coeffiecients ]

    Input:
    S --         Spectrum
    Peaks --     Output from peak finder
    plotcoeff -- True if plot coefficients rather than power
    plotci    -- True if plot confidence limits
    plotentropy  True if plot entropy rather than Wav Power spectrum
    normtovar -- True is normalise to variance
    useLatex  -- True if use latex for labelling
    usewaveletscale -- True if wavelet scale rather than fourier
    scaling   -- "linear" or "log"
    title     -- Title to use for plot
    outfilename  Output file
    maxspect  -- maximum for spectrum plot
    limits_scale limits to scale axis
    clip_fraction -- fraction of max coefficient/power used as clip point in plot 
    ylabels  --  add y labels to image plot 
    """


    # NOTE: 26/6/2013:remove title space!
    
    # Heavily tweaked for figs for proposed 2013 paper !!
    # plotlo,plothi limits of plot
    # ROIlo,ROIhi limits of highlighted area
    plotlo=plotlim[0]
    plothi=plotlim[1]
    ROIlo=int(S.x[0]/0.01+0.01)
    ROIhi=int(S.x[-1]/0.01+0.01)
    print("ROIlo,Ni from x:", ROIlo,ROIhi,S.x[0],S.x[-1])
    ROI=(ROIlo,ROIhi)
    
    fitpeaks=(Peaks is not None)
    # get file name
    # this fails if run from emacs !!
    ####thisfilename=inspect.getsourcefile(inspect.currentframe())
    ####thisfilename=thisfilename.split("\\")[-1]
    thisfilename="Plotter.py"
    # get date and time as a string
    now=datetime.datetime.now().ctime()


    #fig=plt.figure(1,((3.375*2),(3.375*2)*6./8.))  # prl
    fig=plt.figure(1,(8.,6.))  
    plt.clf()
    #ax=plt.subplot2grid((7,3),(3,1),rowspan=4,colspan=2)
    #ax2=plt.subplot2grid((7,3),(0,1),rowspan=3,colspan=2)
    #ax3=plt.subplot2grid((7,3),(3,0),rowspan=4,colspan=1)
    #plt.subplots_adjust(hspace=0.40,right=0.97)
    
    # subplot -- Wavelet 2-d plot
    ax=plt.axes([0.55,0.11,0.4,0.45])#
    plot_wavelet_coefficients(ax,S,plotcoeff,plotlim,ROI,limits_scale=limits_scale,
                              usewaveletscale=usewaveletscale, scaling=scaling,
                              clip_fraction=clip_fraction,ylabels=ylabels)
    # label with filename, date/time
    #plt.text(0.2,-0.18,now,fontsize=8,transform=ax.transAxes,alpha=0.5)
    #plt.text(0.7,-0.18,thisfilename,fontsize=8,transform=ax.transAxes,alpha=0.5)

    # subplot -- original spectrum
    print("PLOT 2")
    ax2=plt.axes([0.55,0.59,0.4,0.36])
    if title is not None:
        if title[0] != "!":
            if useLatex:
                ax2.set_title(r"\verb|%s|"%(title,))
            else:
                ax2.set_title(title)
        else:
            plt.text(0.2,0.80,title[1:],transform=fig.transFigure
                     )
    plot_crosssection_data(ax2,S,plotlim,ROI, maxspect)

    # subplot -- wavelet power/entropy
    print("PLOT 3")
    #ax3=plt.axes([0.1,0.1,0.4,0.45])
    ax3=plt.axes([0.17,0.11,0.35,0.45])
    plot_scale_spectrum_rotated(ax3, S, Peaks, plotci, plotentropy, normtovar,
                                scaling, usewaveletscale,limits_scale)
    ###plt.text(0.02,0.90,r"\textbf{(c)}",transform=ax3.transAxes)
    if outfilename: plt.savefig(outfilename)
    # end

    return plotlo,plothi,ROIlo,ROIhi

pdebug=True
def plot_wavelet_coefficients(ax,
                              S,
                              plotcoeff,
                              plotlim,ROI,
                              usewaveletscale=True,
                              scaling=None,
                              limits_scale=None,
                              clip_fraction=1.0,
                              ylabels=False):
    
    """
    Plot wavelet 2d coefficients
    
    ax               -- matplotlib axes object in which to plot
    S                -- Spectrum object of data
    plotcoeff        -- True if coefficient plot
    usewaveletscale  -- True if wavelet scale
    scaling          -- 'linear or' 'log'
    limits_scale     -- scale range for plot
    limits_scale limits to scale axis
    clip_fraction -- fraction of max coefficient/power used as clip point in plot 
    ylabels  --  add y labels to image plot 
    """
    
    plotlo=plotlim[0]
    plothi=plotlim[1]
    ROIlo=ROI[0]
    ROIhi=ROI[1]
    ax.set_xlabel('Excitation Energy (MeV)')
    ax.yaxis.set_ticklabels(["",""])
    # get data limits from S
    y=S.y
    x=S.x
    print("fix",1,scaling)
    ##x=np.arange(0.0,30.0,0.01) #    FIX
    # if limits specified on scale axis, check if consistent;
    # otherwise take from data (y)
    if limits_scale is not None:
        print("ADJUST LIMITS")
        ylo,yhi=limits_scale
        for lyhi in range(len(y)):
            if S.y[lyhi]-yhi>0: break
        for lylo in range(len(y)):
             if S.y[lylo]-ylo>0: break
        print(lylo,lyhi,S.y[lylo],S.y[lyhi])
        ylo=S.y[lylo]
        yhi=S.y[lyhi]
    else:
        lyhi=len(y)-1
        lylo=0
        ylo=S.y[0]
        yhi=S.y[-1]
    if pdebug: print("lohi", ylo, yhi,y[-1],y[0],lylo,lyhi)

    # setup for 2d plot
    clipfraction=0.94  # targetbase 16
    #clipfraction=0.7  # targetbase 13
    clipfraction=clip_fraction
    cmap=plt.cm.viridis
    cmap=plt.cm.inferno
    # plot  (if) (abs) of coefficients or (else) squared
    if plotcoeff:
        print("PLOT COEFF", plotcoeff)

        cwt=S.cw.getdata()
        maxcwt=np.amax(np.abs(cwt[:,ROIlo:ROIhi].real))
        newcwt=np.clip(np.abs(cwt.real), 0., maxcwt*clipfraction)
        impwr=alphablend_image(newcwt, cmap, maxcwt*clipfraction, plotlo,plothi, ROIlo,ROIhi )
        print("shape impwr",np.shape(impwr),lylo,lyhi,plotlo,plothi)
        ax.imshow(impwr[lylo:lyhi,plotlo:plothi],cmap=cmap,extent=[x[0],x[-1],ylo,yhi],aspect='auto',origin='lower')
        #ax.imshow(impwr,cmap=cmap,extent=[x[0],x[-1],yhi,ylo],aspect="auto")
        #ax.contour(S.pwr,origin='image',extent=[x[0],x[-1],y[0],y[-1]])
        #colorbar()
    else:
        # get coefficients
        print("PLOT POWER")
        cwt=S.cw.getdata()
        # square them
        pwr=(cwt*cwt.conjugate()).real
        # if power is scaled for bias removal, do it
        if S.scalepwr=='scale':
            pwr=np.sqrt(pwr/S.scales[:,np.newaxis])
        else:
            pwr=np.sqrt(pwr)
        # clip to suitable range
        maxcwt=np.amax(pwr)
        pwr=np.clip(pwr, 0., maxcwt*clipfraction)
        maxcwt=np.amax(pwr[:,ROIlo:ROIhi])
        if pdebug: print(ROIlo,ROIhi,plotlo,plothi,np.shape(pwr))
        impwr=alphablend_image(pwr, cmap, maxcwt*clipfraction, plotlo,plothi, ROIlo,ROIhi )
        ax.imshow(impwr[lylo:lyhi,plotlo:plothi],cmap,extent=[x[0],x[-1],ylo,yhi],aspect="auto",origin='lower')
        #ax.contour(pwr[:,ROIlo:ROIhi],origin='image',extent=[x[0],x[-1],y[0],y[-1]],cmap=cmap)

    # set limits and axis decorations
    ax.set_xlim(x[0],x[-1])
    if pdebug: print("lohi", ylo, yhi,y[-1],y[0],ylabels)
    # handle log scaling of scale axis
    if scaling=="log" or scaling=="linlog":
        # matplotlib 2.0 has true log scaling of images but that breaks
        # the way we do things -- now have to constrauct log axes ourselves
        # without using true log scaling. All done with ticks, but have to
        # be very careful with scales.
        if pdebug: print("scaling >>", scaling)
        #locator=mpl.ticker.LogLocator()
        #ax.yaxis.set_major_locator(locator)
        #ax.yaxis.set_minor_locator(locator)
        ticks=np.array([0.001,0.01,0.1,1.0,10.0,100.0])
        l,h=getlimits(ylo,yhi,ticks)
        sc=(yhi-ylo)/(np.log10(yhi)-np.log10(ylo))
        ticks=ylo+sc*(np.log10(ticks[l:h])-np.log10(ylo))
        print("ticks->",ticks)
        print(ylo,yhi,y[lylo],y[lyhi],y[0],y[-1])
        ax.yaxis.set_ticks(ticks)
        minorticks=np.array([0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
                             0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,
                             0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
                             2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,
                             20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0])
        l,h=getlimits(ylo,yhi,minorticks)
        minorticks=ylo+sc*(np.log10(minorticks[l:h])-np.log10(ylo))
        ax.yaxis.set_ticks(minorticks,minor=True)
        ax.set_ylim(ylo,yhi)  # must follow all the reticking
    ###plt.text(0.91,0.9,r"\textbf{(b)}",transform=ax.transAxes)

def getlimits(ylo,yhi,tickarray):
    for lo in range(len(tickarray)):
        if tickarray[lo]-ylo>0: break
    for hi in range(len(tickarray)):
        if tickarray[hi]-yhi>0: break
    return lo,hi
   

def alphablend_image(data, cmap, maxdata, plotlo,plothi, ROIlo,ROIhi ):
    """
    Here we have to do alpha blending of end fading ourselves
    because the pdf backend can't handle it! (png can).
    """
    # normalise color map to data limits
    imnorm=mpl.colors.Normalize(0.0,maxdata)
    impwr=plt.get_cmap(cmap)(imnorm(data))
    # blend ends
    #impwr[:,plotlo:ROIlo,:3]=(impwr[:,plotlo:ROIlo,:3]*0.5+1.0)/(1.0+0.5)
    impwr[:,plotlo:ROIlo,:3]=(impwr[:,plotlo:ROIlo,:3]*0.8+0.2)/(0.2+0.8)
    impwr[:,ROIhi:plothi,:3]=(impwr[:,ROIhi:plothi,:3]*0.8+0.2)/(0.2+0.8)
    ls=LightSource(130.0,20.0)
    #impwr=ls.shade_rgb(impwr,data)
    return impwr


def plot_crosssection_data(ax2,S,plotlim,ROI, maxspect=None):
    ###plt.text(0.91,0.9,r"\textbf{(a)}",transform=ax2.transAxes)

    plotlo=plotlim[0]
    plothi=plotlim[1]
    ROIlo=ROI[0]
    ROIhi=ROI[1]
    # find max value in limits
    maxcs=np.amax(S.basicdata[plotlo:plothi])
    print("maxcs",maxcs)
    lmaxcs=np.log10(maxcs)*10.0
    lmaxcs=float(np.rint(lmaxcs))/10.0
    maxcs=10.0**lmaxcs
    print(lmaxcs,maxcs,maxspect)
    # FIX - move y label to input
    ax2.set_ylabel(r'Counts/chan')
    #ax2.set_ylabel(r'\noindent B(E1) strength\\[5pt]\phantom{xx}(arb. units)')
    #ax2.set_ylabel(r'$d^2\sigma/d\Omega dE$ [mb/sr/MeV]')
    plt.setp(ax2.get_xticklabels(), visible=False)

    # ========================================
    # convert to counts per ch for paper
    # factor = conversion mb -> counts/mb from data/factor.dat
    # ========================================
    #factor = 1.32773e4
    x=S.x
    x=np.arange(0.0,30.0,0.01) #    FIX
    print("++++++++++",np.shape(x),plotlo,plothi)
    ax2.plot(x[plotlo:plothi],S.basicdata[plotlo:plothi],'k-')
    ax2.set_xlim(x[plotlo],x[plothi])
    if maxspect is not None:
        ax2.set_yticks([0.0,maxspect])
        ax2.set_ylim(0.0,maxspect)
        #ax2.set_yticklabels(["0","1"])
    else:
        ax2.set_ylim(0.0,maxcs)
    plt.axvline(x[ROIlo],color='0.4',lw=1.5)
    plt.axvline(x[ROIhi],color='0.4',lw=1.5)


def plot_scale_spectrum_rotated(ax3, S, Peaks=None, plotci=False, plotentropy=False,
                               normtovar=True,  scaling="log", usewaveletscale=True,
                               limits_scale=None):
    import pylab as plt
    fitpeaks=(Peaks is not None)
    if usewaveletscale:
        ax3.set_ylabel('Wavelet scale (MeV)')
    else:
        ax3.set_ylabel('Fourier scale (MeV)')

    if plotentropy:
        ax3.set_ylabel('Entropy index (arb. units)')
        ax3.plot(S.y,S.entindex,'b-')
    else:
        ax3.set_xlabel('Wavelet power (arb. units)')
        if pdebug: print(np.shape(S.scalespec),np.shape(S.y))
        ss=S.scalespec[::-1]
        if normtovar:
            vara=S.vara
        else:
            vara=1.0
        if pdebug: print("scaling=>",scaling)
        # note: norm to scale (scalepwr) is done in S
        #temp:
        if scaling=='linear':
            selectedplot=ax3.semilogx
        elif scaling=='linlog':
            selectedplot=ax3.semilogy
        elif scaling=='linlin':
            selectedplot=ax3.plot
        else:
            selectedplot=ax3.loglog
        selectedplot(S.scalespec/vara+0.01,S.y,'k-')
        y=S.y
        if limits_scale is not None:
            ylo,yhi=limits_scale
            for lyhi in range(len(y)):
                if y[lyhi]-yhi<0: break
            for lylo in range(len(y)):
                if y[lylo]-ylo>0: break
        else:
            ylo=y[0]
            yhi=y[-1]
        iy=0
        ax3.set_ylim(ylo,yhi)#y[iy])   # FIX 
        #ylo=0
        if pdebug: print("y->",y[iy],y[0],y[-1])
        svmax=S.scalespec[0]/vara
        svmin=S.scalespec[0]/vara
        for i,yval in enumerate(y):
            if ylo <= yval <= yhi:
                sval=S.scalespec[i]/vara
                if sval > svmax: svmax = sval
                if sval < svmin: svmin = sval
        if S.scalepwr=='scale':
            ax3.set_xlim(1000.0,20.0)
            ax3.set_xlim(svmax,svmin)
        else:
            ax3.set_xlim(1000.0,0.01)
            ax3.set_xlim(200.0,0.1)  # 28/11/2013 hack
            ax3.set_xlim(svmax,svmin)
            if scaling=='linear' or scaling=='log':
                ax3.set_xticklabels(["","0.1","1","10","100"])
        xhi,xlo=ax3.get_xlim()
        ax3.tick_params(which='both',right='on')
        # add peak identification to plot
        if fitpeaks:
            for p,ok,pamp in zip(Peaks.peaks,Peaks.peaksig,Peaks.peakamplitude):
                if pdebug: print('peak',p,pamp,vara)
                if p>yhi or p<ylo: continue
                pamp=pamp/vara
                if scaling=='linear' or scaling=='log':
                    xp=(np.log10(pamp)-np.log10(xlo))/(np.log10(xhi)-np.log10(xlo))
                else:
                    xp=(pamp-xlo)/(xhi-xlo)
                if scaling=='log' or scaling=='linlog':
                    yp=(np.log10(p)-np.log10(ylo))/(np.log10(yhi)-np.log10(ylo))
                else:
                    yp=(p-ylo)/(yhi-ylo)
                xp=1.0-xp
                pax=(xp,yp)
                if ok:
                    ax3.arrow(0.99,pax[1],-(0.88-pax[0]),0.0,transform=ax3.transAxes,clip_on=False,color='0.5',head_width=0.03,head_starts_at_zero=True)
            # add a peak  or 2...
            k0=3.5 # why not 5 ???????????????? notes of 28/6/12
            #if pdebug:print("File", "RTBA")
            if pdebug:print("peak position", "strength", "significant")
            P0=np.zeros(len(S.y))
            for p,pamp,ok in zip(Peaks.peaks, Peaks.peakamplitude,Peaks.peaksig):
                Pi=np.exp(-(S.y-p)**2/(p/k0)**2)*pamp/vara
                P0+=Pi
                print("{0:6.3f}   {1:5.2f}  {2}".format(p,peakintegral(Pi,S.y),ok))
            #########ax3.loglog(P0, S.y, 'g-') # plot peaks
        if plotci:
            # plot confidence limits
            selectedplot(S.cipn+0.001, S.y, 'm-',lw=1.0)
            selectedplot(S.Pkpn+0.001, S.y, 'r-',lw=1.0)


def peakintegral(peak, s ):
    """
    integrate peaks over scale : must take change
    of variable into account.
    i.e. scale = 1/k, and integral over k is related
    to strength
    """
    ignd=peak/s/s
    I=sum(((ignd[0:-1]+ignd[1:])/2)*(s[1:]-s[:-1]))
    return I

def plot_scale_spectrum(S, Peaks=None, plotci=False, normtovar=True, scaling="log",
                        usewaveletscale=True, outfilename=None):
    import pylab as plt
    fitpeaks=(Peaks is not None)
    plt.figure(2,figsize=(5.5,4))
    plt.clf()
    if usewaveletscale:
        plt.xlabel('Wavelet scale (MeV)')
    else:
        plt.xlabel('Fourier scale (MeV)')
    plt.ylabel('Wavelet power (arb. units)')
    #plt.title(shorttargetname+" "+angle)
    if pdebug: print("normtovard",normtovar)
    if normtovar:
        vara=S.vara
    else:
        vara=1.0
    #print(S.scalepwr,normtovar,S.vara)
    #S.scalepwr=None
    ax=plt.gca()
    if scaling=='linear':
        selectedplot=ax.semilogx
    elif scaling=='linlog':
        selectedplot=ax.semilogy
    elif scaling=='linlin':
        selectedplot=ax.plot
    else:
        selectedplot=ax.loglog
    print("vara",vara)

    selectedplot(S.y,S.scalespec/vara+0.01,'k-')
       
    plt.xlim(S.y[0],S.y[-1])
    #plt.xlim(S.y[0],2.0)
    if S.scalepwr=='scale':
        selectedplot(S.xfft, S.pwrdata/S.xfft/vara+0.01,'g-')
    else:
        selectedplot(S.xfft, S.pwrdata/vara+0.01,'g-')

    if fitpeaks:
        for p, ok in zip(Peaks.peaks, Peaks.peaksig):
            if ok: plt.axvline(p,)
            else: plt.axvline(p,alpha=0.2)
        k0=3.5
        P0=np.zeros(len(S.y))
        for p,pamp,ok in zip(Peaks.peaks, Peaks.peakamplitude,Peaks.peaksig):
            Pi=np.exp(-(S.y-p)**2/(p/k0)**2)*pamp/vara
            P0+=Pi
    plt.tight_layout()
    if outfilename: plt.savefig(outfilename)

def set_plot_text_method(useLatex):
    import pylab as plt
    if useLatex:
        # set up Latex-style plot labelling
        plt.rc('text',usetex=True)
        plt.rc('figure.subplot',top=0.94, right=0.95)
        plt.rc('legend',fontsize='small')
        #plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
        #plt.rc('font',**{'family':'serif','serif':['stix']})
        #plt.rc('font',**{'family':'sans-serif','sans-serif':['stixsans']})
        plt.rc('font',  size=20.0)
        plt.rc('text.latex',preamble=[r'\usepackage[T1]{fontenc}',r'\usepackage{cmbright}'])


if  __name__=="__main__":
    print("Version of 26/6/2017")

    from SpectrumAnalyser8 import Spectrum, PeakFinder
    import pylab as plt
##    import os
##    import sys

    # run control (bool) and user settings
    printer=0         # print plots to a file
    plotcoeff=False       # plot coefficients or power
    plotsmooth=0      # show smoothed spectrum over spectrum
    plotentropy=0     # plot entropy index or scale spectrum
    usesmooth=0       # use spectrum-smoothed
    plotspectrum=1    # plot spectrum or autocorrelation
    normtovar=1       # normalize power spectrum to variance
    usederiv=0
    waveletclass=Morlet
    order=2
    fitpeaks=True
    plotci=False
    outputscalespectrum=True
    usewaveletscale=False
    useLatex=True
    doshoulders=False



    datafile="../../uct2013c/pb208/pb208gdr.dat"
    datafile="TEST2"
    sig=0.03/2.35 # 50 keV
    sigsm=0.0   # width of smoothing gaussian (MeV)
    Nf=512            # channels in FFT for power spectrum
    maxscale=4        # maximum scale is Ns/maxscale
    notes=16          # notes per octave. 0 gives linear scale

    # region of interest is gdr
    Nlo=1000
    Nhi=2000
    scaling='linear'
    #scaling='log'

    set_plot_text_method(useLatex)

    # read in data and do wavelet transform
    sigsm=0.0
    if usewaveletscale:
        scaletype="wavelet"
    else:
        scaletype="fourier"
    #Nlo=900
    #Nhi=1500
    #Mlo=800
    #Mhi=1900
    ROIlo=900
    ROIhi=1500
    plotlo=800
    plothi=1900
    #datafile="TEST1"
    S=Spectrum(waveletclass, datafile, Nlo, Nhi, sig, 0, sigsm,order=order,scaling=scaling,scaletype=scaletype)  # 28/11/2013 hack
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
    ##     Peaks.peaks.append(0.598)
    ##     Peaks.peakamplitude.append(2305.)
    ##     Peaks.peaksig.append(True)
    ##     #8/5/2014:
    ##     Peaks.peaks.append(0.11)
    ##     Peaks.peakamplitude.append(10000.)
    ##     Peaks.peaksig.append(True)
    ## else: Peaks=None

    # Plots ------------------------
    title=r"!\Huge\noindent$^{208}$Pb"
    set_plot_text_method(useLatex)
    plot_triple(S, Peaks=None, plotlim=(Nlo,Nhi),plotcoeff=plotcoeff, plotci=plotci,
                plotentropy=plotentropy, normtovar=normtovar,
                useLatex=useLatex, usewaveletscale=usewaveletscale,
                scaling=scaling, title=title,
                outfilename=None,maxspect=None,
                limits_scale=(0.05,2.0))
    plotfile=datafile[:-8]+".pdf"
    print(plotfile)
    #plt.savefig(plotfile)
    #plt.savefig("corrected2_fig4.pdf")
    plot_scale_spectrum(S, Peaks=Peaks, plotci=plotci, normtovar=normtovar,
                         usewaveletscale=usewaveletscale,  outfilename=None)
    plt.figure(3)
    plot_wavelet_coefficients(plt.gca(),
                              S,
                              plotcoeff,
                              plotlim=(Nlo,Nhi),ROI=(Nlo,Nhi),
                              usewaveletscale=False,
                              scaling=scaling,
                              limits_scale=None,
                              clip_fraction=1.0,
                              ylabels=True)

    plt.show()


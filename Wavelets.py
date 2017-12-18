import numpy as NP

"""
A module which implements the continuous wavelet transform

Wavelet classes:
Morlet
MorletReal
MexicanHat
Paul2      : Paul order 2
Paul4      : Paul order 4
DOG1       : 1st Derivative Of Gaussian
DOG4       : 4th Derivative Of Gaussian
Haar       : Unnormalised version of continuous Haar transform
HaarW      : Normalised Haar

Usage e.g.
wavelet=Morlet(data, largestscale=2, notes=0, order=2, scaling="log")
 data:  Numeric array of data (float), with length ndata.
        Optimum length is a power of 2 (for FFT)
        Worst-case length is a prime
 largestscale:
        largest scale as inverse fraction of length
        scale = len(data)/largestscale
        smallest scale should be >= 2 for meaningful data
 notes: number of scale intervals per octave
        if notes == 0, scales are on a linear increment
 order: order of wavelet for wavelets with variable order
        [Paul, DOG, ..]
 scaling: "linear" or "log" scaling of the wavelet scale.
        Note that feature width in the scale direction
        is constant on a log scale.
        
Attributes of instance:
wavelet.cwt:       2-d array of Wavelet coefficients, (nscales,ndata)
wavelet.nscale:    Number of scale intervals
wavelet.scales:    Array of scale values
                   Note that meaning of the scale will depend on the family
wavelet.fourierwl: Factor to multiply scale by to get scale
                   of equivalent FFT
                   Using this factor, different wavelet families will
                   have comparable scales

References:
A practical guide to wavelet analysis
C Torrance and GP Compo
Bull Amer Meteor Soc Vol 79 No 1 61-78 (1998)
naming below vaguely follows this.

updates:
(24/2/07):  Fix Morlet so can get MorletReal by cutting out H
(10/04/08): Numeric -> numpy
(25/07/08): log and lin scale increment in same direction!
            swap indices in 2-d coeffiecient matrix
            explicit scaling of scale axis
(17/11/2010): Implement CWT of derivative of spectrum by multiplying
             fft of data by i.omega -- add argument to constructor.
"""

class Cwt(object):
    """
    Base class for continuous wavelet transforms
    Implements cwt via the Fourier transform
    Used by subclass which provides the method wf(self,s_omega)
    wf is the Fourier transform of the wavelet function.
    Returns an instance.

    Parameters
    ----------
    data:  ndarray of float
        Numpy array of data (float), with length ndata.
        Optimum length is a power of 2 (for FFT)
        Worst-case length is a prime
    largestscale: int
        largest scale as inverse fraction of length.
        scale = len(data)/largestscale.
        smallest scale should be >= 2 for meaningful data.
    notes: int
        number of scale intervals per octave.
        if notes == 0, scales are on a linear increment.
    order: int
        order of wavelet for wavelets with variable order
        [Paul, DOG, ..]
    scaling: str
        "linear" or "log" scaling of the wavelet scale.
        Note that feature width in the scale direction
        is constant on a log scale.
    deriv : bool
        If True, take numerical derivative of data before transform.
    """

    fourierwl=1.00

    def _log2(self, x):
        """ utility function to return (integer) log2

        Parameters
        ----------
        x : float
            Input value

        Returns
        -------
        int: log to base 2 of `x`
        """
        return int( NP.log(float(x))/ NP.log(2.0)+0.0001 )

    def __init__(self, data, largestscale=1, notes=0, order=2, scaling='linear', deriv=False):
        """
        Continuous wavelet transform of data

        data:    data in array to transform, length must be power of 2
        notes:   number of scale intervals per octave
        largestscale: largest scale as inverse fraction of length
                 of data array
                 scale = len(data)/largestscale
                 smallest scale should be >= 2 for meaningful data
        order:   Order of wavelet basis function for some families
        scaling: Linear or log
        """
        ndata = len(data)
        self.order=order
        self.largestscale=largestscale
        #print(ndata,largestscale)
        self._setscales(ndata,largestscale,notes,scaling)
        self.cwt= NP.zeros((self.nscale,ndata), NP.complex64)
        omega= NP.array([x for x in range(0,ndata//2)]+[x for x in range(-ndata//2,0)])*(2.0*NP.pi/ndata)
        datahat=NP.fft.fft(data)
        self.fftdata=datahat
        #self.psihat0=self.wf(omega*self.scales[3*self.nscale/4])
        # loop over scales and compute wvelet coeffiecients at each scale
        # using the fft to do the convolution
        for scaleindex in range(self.nscale):
            currentscale=self.scales[scaleindex]
            self.currentscale=currentscale  # for internal use
            s_omega = omega*currentscale
            psihat=self.wf(s_omega)
            psihat = psihat *  NP.sqrt(2.0*NP.pi*currentscale)
            convhat = psihat * datahat 
            if deriv: convhat *= 1J*omega
            W    = NP.fft.ifft(convhat)
            self.cwt[scaleindex,0:ndata] = W 
        return
    
    def _setscales(self,ndata,largestscale,notes,scaling):
        """
        if notes non-zero, returns a log scale based on notes per ocave
        else a linear scale
        (25/07/08): fix notes!=0 case so smallest scale at [0]
        """
        if scaling=="log":
            if notes<=0: notes=1 
            # adjust nscale so smallest scale is 2 
            noctave=self._log2( ndata//largestscale//2 )
            self.nscale=notes*noctave
            self.scales=NP.zeros(self.nscale,float)
            for j in range(self.nscale):
                self.scales[j] = ndata/(self.largestscale*(2.0**(float(self.nscale-1-j)/notes)))
        elif scaling=="linear":
            nmax=ndata//largestscale//2
            self.scales=NP.arange(float(2),float(nmax))
            self.nscale=len(self.scales)
        else: raise ValueError("scaling must be linear or log")
        return
    
    def getdata(self):
        """returns wavelet coefficient array
        
        """
        return self.cwt
    def getcoefficients(self):
        return self.cwt
    def getpower(self):
        """
        returns square of wavelet coefficient array
        """
        return (self.cwt* NP.conjugate(self.cwt)).real
    def getscales(self):
        """
        returns array containing scales used in transform
        """
        return self.scales
    def getnscale(self):
        """
        return number of scales
        """
        return self.nscale

# wavelet classes    
class Morlet(Cwt):
    """
    Morlet wavelet
    """
    _omega0=5.0
    fourierwl=4* NP.pi/(_omega0+ NP.sqrt(2.0+_omega0**2))
    def wf(self, s_omega):
        H= NP.ones(len(s_omega))
        n=len(s_omega)
        for i in range(len(s_omega)):
            if s_omega[i] < 0.0: H[i]=0.0
        # !!!! note : was s_omega/8 before 17/6/03
        xhat=0.75112554*( NP.exp(-(s_omega-self._omega0)**2/2.0))*H
        return xhat

class MorletReal(Cwt):
    """
    Real Morlet wavelet
    """
    _omega0=5.0
    fourierwl=4* NP.pi/(_omega0+ NP.sqrt(2.0+_omega0**2))
    def wf(self, s_omega):
        H= NP.ones(len(s_omega))
        n=len(s_omega)
        for i in range(len(s_omega)):
            if s_omega[i] < 0.0: H[i]=0.0
        # !!!! note : was s_omega/8 before 17/6/03
        xhat=0.75112554*( NP.exp(-(s_omega-self._omega0)**2/2.0)+ NP.exp(-(s_omega+self._omega0)**2/2.0)- NP.exp(-(self._omega0)**2/2.0)+ NP.exp(-(self._omega0)**2/2.0))
        return xhat
    
class Paul4(Cwt):
    """
    Paul m=4 wavelet
    """
    fourierwl=4* NP.pi/(2.*4+1.)
    def wf(self, s_omega):
        n=len(s_omega)
        xhat= NP.zeros(n)
        xhat[0:n//2]=0.11268723*s_omega[0:n//2]**4* NP.exp(-s_omega[0:n//2])
        #return 0.11268723*s_omega**2*exp(-s_omega)*H
        return xhat

class Paul2(Cwt):
    """
    Paul m=2 wavelet
    """
    fourierwl=4* NP.pi/(2.*2+1.)
    def wf(self, s_omega):
        n=len(s_omega)
        xhat= NP.zeros(n)
        xhat[0:n//2]=1.1547005*s_omega[0:n//2]**2* NP.exp(-s_omega[0:n//2])
        #return 0.11268723*s_omega**2*exp(-s_omega)*H
        return xhat

class Paul(Cwt):
    """
    Paul order m wavelet
    """
    def wf(self, s_omega):
        Cwt.fourierwl=4* NP.pi/(2.*self.order+1.)
        m=self.order
        n=len(s_omega)
        normfactor=float(m)
        for i in range(1,2*m):
            normfactor=normfactor*i
        normfactor=2.0**m/ NP.sqrt(normfactor)
        xhat= NP.zeros(n)
        xhat[0:n//2]=normfactor*s_omega[0:n//2]**m* NP.exp(-s_omega[0:n//2])
        #return 0.11268723*s_omega**2*exp(-s_omega)*H
        return xhat

class MexicanHat(Cwt):
    """
    2nd Derivative Gaussian (mexican hat) wavelet
    """
    fourierwl=2.0* NP.pi/ NP.sqrt(2.5)
    def wf(self, s_omega):
        # should this number be 1/sqrt(3/4) (no pi)?
        #s_omega = s_omega/self.fourierwl
        #print max(s_omega)
        a=s_omega**2
        b=s_omega**2/2
        return a* NP.exp(-b)/1.1529702
        #return s_omega**2*exp(-s_omega**2/2.0)/1.1529702

class DOG4(Cwt):
    """
    4th Derivative Gaussian wavelet
    see also T&C errata for - sign
    but reconstruction seems to work best with +!
    """
    fourierwl=2.0* NP.pi/ NP.sqrt(4.5)
    def wf(self, s_omega):
        return s_omega**4* NP.exp(-s_omega**2/2.0)/3.4105319

class DOG1(Cwt):
    """
    1st Derivative Gaussian wavelet
    but reconstruction seems to work best with +!
    """
    fourierwl=2.0* NP.pi/ NP.sqrt(1.5)
    def wf(self, s_omega):
        dog1= NP.zeros(len(s_omega),complex64)
        dog1.imag=s_omega* NP.exp(-s_omega**2/2.0)/sqrt(pi)
        return dog1

class DOG(Cwt):
    """
    Derivative Gaussian wavelet of order m
    but reconstruction seems to work best with +!
    """
    def wf(self, s_omega):
        try:
            from scipy.special import gamma
        except ImportError:
            print( "Requires scipy gamma function")
            raise ImportError
        Cwt.fourierwl=2* NP.pi/ NP.sqrt(self.order+0.5)
        m=self.order
        dog=1.0J**m*s_omega**m* NP.exp(-s_omega**2/2)/ NP.sqrt(gamma(self.order+0.5))
        return dog

class Haar(Cwt):
    """
    Continuous version of Haar wavelet
    """
    #    note: not orthogonal!
    #    note: s_omega/4 matches Lecroix scale defn.
    #          s_omega/2 matches orthogonal Haar
    # 2/8/05 constants adjusted to match artem eim

    fourierwl=1.0#1.83129  #2.0
    def wf(self, s_omega):
        haar= NP.zeros(len(s_omega),complex64)
        om = s_omega[:]/self.currentscale
        om[0]=1.0  #prevent divide error
        #haar.imag=4.0*sin(s_omega/2)**2/om
        haar.imag=4.0* NP.sin(s_omega/4)**2/om
        return haar

class HaarW(Cwt):
    """
    Continuous version of Haar wavelet (norm)
    """
    #    note: not orthogonal!
    #    note: s_omega/4 matches Lecroix scale defn.
    #          s_omega/2 matches orthogonal Haar
    # normalised to unit power

    fourierwl=1.83129*1.2  #2.0
    def wf(self, s_omega):
        haar= NP.zeros(len(s_omega),complex64)
        om = s_omega[:]#/self.currentscale
        om[0]=1.0  #prevent divide error
        #haar.imag=4.0*sin(s_omega/2)**2/om
        haar.imag=4.0* NP.sin(s_omega/2)**2/om
        return haar


if __name__=="__main__":
    print( "Extracted from cw-dieter2.py (30/7/09)")

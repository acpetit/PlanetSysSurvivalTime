import numpy as np
from numpy import pi

from laplacecoefficients import b_lap,db_lap
import survivaltime as st

from scipy.optimize import fsolve


def overlap_plane(Nu12,Nu23,masses=1e-5*np.ones(3),kmax=200,verbose=False,exact_lap=False):
    """
    Compute the number of resonances at each point of a nu12, nu23 plane
    
    Inputs:
    Nu12,Nu23 is a meshgrid  2D array of period ratios
    masses are the three planets masses
    kmax is the maximum p+q value over which the sum is done
    verbose : bool to print or not evolution in the computation
    exact_lap : using the exact Laplace coefficients instead of the excellent approximation B.6.

    Return 2D array of the number of resonances passing by the considered pixel
    """

    

    #See eqs. (21,22)
    Eta = Nu12*(1-Nu23)/(1-Nu12*Nu23)
    Nu = 1/(1/(1/Nu12-1)+1/(1-Nu23))

    #Results
    Overlap = np.zeros_like(Nu12)

    for k in range(2,kmax): #Indexes starts at 2
        #Compute grid with p and q for given k=p+q
        Ps = np.rint(k*Eta)
        Ps0 = Ps.copy()

        #Correction for side effects
        Ps += Ps==0
        Ps -= Ps==k
        #Eta at the closest resonance
        Eta0 = Ps/k
        Dist = Eta-Eta0
        Qs = k-Ps
        
        #Find resonances loci
        Nu12_res = Nu12.copy()
        Nu23_res = Nu23.copy()
        if np.max(abs(Dist))>0.01:
            j=0
            dEta = Nu12_res*(1-Nu23_res)/(1-Nu12_res*Nu23_res)-Eta0
            while (j<10) and (np.max(abs(dEta[(Ps0>0)&(Ps0<k)]))>1e-3):
                if verbose:
                    print('Gradient :',k,j)
                j+=1
                dEta = Nu12_res*(1-Nu23_res)/(1-Nu12_res*Nu23_res)-Eta0

                GrEta1 = (1-Nu23_res)/(1-Nu12_res*Nu23_res)**2
                GrEta2 = -Nu12_res*(1-Nu12_res)/(1-Nu12_res*Nu23_res)**2

                Nu12_res -= 0.5*dEta/GrEta1 #Unstable without the 0.5
                Nu23_res -= 0.5*dEta/GrEta2

        if verbose:
            print('Width :',k)

        #Width computed on the resonance in term of eta (eq. 55)
        if exact_lap:
            Widths = _res_width_ex(Ps,k,Nu12_res,Nu23_res,masses)/2
        else:
            Widths = _res_width_ex_fast(Ps,k,Nu12_res,Nu23_res,masses)/2

        Overlap += abs(Dist)<Widths
    return Overlap

def nu_overlaplimit(etas,masses,m0=1.):
    nuguess = 0.05*np.ones_like(etas)
    nuov = fsolve(_fillingfactor_th,nuguess,args=(etas,masses,m0))
    return nuov

def _fillingfactor_th(nu,eta,masses,m0=1.):
    "Auxiliary function to compute theoretical overlap limit"
    m1,m2,m3 = masses
    nu12,nu23 = st.nu_eta_to_nus(nu,eta)
    plsep = 1/(1/(1-nu12**(2/3))+1/(1-nu23**(2/3)))
    Mfac = (m1*m3/m0**2*(eta**2/nu12**(4/3)+1+(1-eta)**2*nu23**(4/3)))**.5

    return np.nan_to_num(Mfac*(38/pi)**.5*4*2**.5/3*(eta*(1-eta))**1.5/plsep**4-1)


def _res_width_ex(Ps,PQ,Nu12_res,Nu23_res,masses,m0 = 1):
    """
    Compute the frequency of the Laplace resonance as a function of the masses and the period ratios
    nu12 = n2/n1
    nu23 = n3/n2
    nu23 is computed at the center as a function of nu12 p and q
    Eq. 42 used for Rpq
    """
    m1,m2,m3 = masses
    eta = Ps/PQ

    al12 = Nu12_res**(2/3)
    al23 = Nu23_res**(2/3)
    plsep = (1-al12)*(1-al23)/(2-al12-al23)
    
    K2red = 3*(1+m2/m1*eta**2/al12**2+(1-eta)**2*al23**2*m2/m3)#Reduced because most of the terms are not needed
    bp = b_lap(.5,Ps,al12)
    bq = b_lap(.5,PQ-Ps,al23)
    dbp = db_lap(.5,Ps,al12)
    dbq = db_lap(.5,PQ-Ps,al12)
    
    
    Rpq = m1*m3/m0**2*al23*(1/(1-Nu23_res)*bq*(bp+al12*dbp)+
                            al23/(Nu12_res**-1-1)*bp*dbq+
                            3*bp*bq/(2*(1-Nu23_res)*(1/Nu12_res-1))) #Same as K2
    Rext = 1/2*((2*Ps+1)*bp+al12*dbp)
    Rint = 1/2*(2*(PQ-Ps)*bq+al23*dbq)
    Nu = 1/(1/(1/Nu12_res-1)+1/(1-Nu23_res))
    Rpq += m1*m3/m0**2*al23*Rext*Rint*(1/(PQ*Nu-1))
    
    freq = np.sqrt(abs(Rpq)*K2red)
    res = np.nan_to_num(4*np.sqrt(2)/3*eta*(1-eta)/plsep*freq)
    return res


def _res_width_ex_fast(Ps,PQ,Nu12_res,Nu23_res,masses,m0 = 1):
    """
    Compute the frequency of the Laplace resonance as a function of the masses and the period ratios
    nu12 = n2/n1
    nu23 = n3/n2
    nu23 is computed at the center as a function of nu12 p and q
    Eq. 42 used for Rpq

    Laplace coefficients computed with the full simplified equivalent (app. B)
    """
    
    m1,m2,m3 = masses
    eta = Ps/PQ
    Qs = PQ-Ps

    al12 = Nu12_res**(2/3)
    al23 = Nu23_res**(2/3)
    plsep = (1-al12)*(1-al23)/(2-al12-al23)
    
    K2red = 3*(1+m2/m1*eta**2/al12**2+(1-eta)**2*al23**2*m2/m3)#Reduced because most of the terms are not needed
    bp = 2*al12**Ps/(pi*Ps*(1-al12**2))**.5
    bq = 2*al23**Qs/(pi*Qs*(1-al23**2))**.5
    dbp = bp/al12*(Ps-1+1/(1-al12**2))
    dbq = bq/al23*(Qs-1+1/(1-al23**2))
    
    Nu = 1/(1/(Nu12_res**-1-1)+1/(1-Nu23_res))
    
    Rpq = m1*m3/m0**2*al23*(1/(1-Nu23_res)*bq*(bp+al12*dbp)+
                            al23/(Nu12_res**-1-1)*bp*dbq+
                            3*bp*bq/(2*(1-Nu23_res)*(1/Nu12_res-1))) #Same as K2
    Rext = 1/2*((2*Ps+1)*bp+al12*dbp)
    Rint = 1/2*(2*Qs*bq+al23*dbq)
    Nu = 1/(1/(1/Nu12_res-1)+1/(1-Nu23_res))
    Rpq += m1*m3/m0**2*al23*Rext*Rint*(1/(PQ*Nu-1))
    
    freq = np.sqrt(abs(Rpq)*K2red)
    res = np.nan_to_num(4*np.sqrt(2)/3*eta*(1-eta)/plsep*freq)
    return res

"""Library of functions used in various jupyter notebook for plotting AMD criteria.
Goal is to have centralized version of those functions and update them only once.

Written by Antoine Petit (antoine.petit@obspm.fr)
"""
#Modules
import numpy as np
from math import pi
import scipy.optimize as optimize

#Few utilities functions 
def AMDpl_to_ecc(C,Lam):
    return np.sqrt(1-(1-C/Lam)**2)


#Functions used to compute the critical AMD based on the collision condition (see Laskar and Petit 2017)
def Cc_col(alpha,gamma,pec=1e-14):
    """Critical AMD for colllision in the secular system (Laskar and Petit 2017)"""
    ec=ecrit(alpha,gamma,pec)
    return (gamma*np.sqrt(alpha)*(1-np.sqrt(1-ec**2))
            +(1-np.sqrt(1-ep(alpha,ec)**2)))

#Critical eccentricity computed with Newton's method
def ecrit(alpha,gamma,pec=1e-14):
    """Compute the critical eccentricity for a collision  
    We build a meshgrid if alpha and gamma are both changing
    
    the result has the shape of alpha
    """
    
    
    ecc=np.zeros_like(alpha)
    necc=ecc-new_step(alpha,gamma,ecc)
    nit=0
    while (np.max(abs(ecc-necc))>pec) and (nit<20):
        nit+=1
        ecc=necc
        necc =np.minimum(1,necc-new_step(alpha,gamma,ecc))

    #Remove extra dimensions
    ecc=necc
    return ecc

def ep(alpha,e):
    return 1-alpha*(1+e)
#On def e0 comme le min de e0 et 1
def e0(alpha):
    return np.minimum(1,1/alpha-1)

def new_step(a,g,e):
    ns=Fe(a,g,e)/dFe(a,g,e)
    #We add these lines to avoid the case of alpha=0 perturbing the computation
    if (np.array(ns).size==1) and np.isnan(ns):
        ns=0
    else:
        ns[np.isnan(ns)]=0
    return ns

def Fe(alpha,gamma,e):
    return alpha*e+gamma*e/np.sqrt(abs(alpha*(1-e**2)+gamma**2*e**2))-1+alpha
def dFe(alpha,gamma,e):
    return alpha + gamma*alpha/abs(alpha*(1-e**2)+gamma**2*e**2)**1.5


### 
def Cc_MMR(alpha,epsilon,gamma):
    """Critical AMD based on MMR overlap (Petit, Laskar & Boue 2017)"""
    crit=mmr_crit(alpha,epsilon)
    return (crit>0)*crit**2/2*gamma*np.sqrt(alpha)/(1+gamma*np.sqrt(alpha))
def mmr_crit(alpha,epsilon):
    r=0.8019857395
    return (3**4*(1-alpha)**5)/(2**9*r*epsilon)-32*r*epsilon/(9*(1-alpha)**2)


###
def mmr_Hadden(alpha,epsilon):
    """MMR overlap criterion proposed by Hadden and Litwick)"""
    return (1-alpha)*np.exp(-2.2*(epsilon/(1-alpha)**4)**(1/3))
def Cc_MMR_Hadden(alpha,epsilon,gamma):
    return gamma*np.sqrt(alpha)/(1+gamma*np.sqrt(alpha))*mmr_Hadden(alpha,epsilon)**2/2


###
def Cc_hill(alpha,epsilon,gamma):
    """Critical AMD based on 2 planets Hill-stability (Petit, Laskar, Boue 2018)"""
    zeta=gamma/(1+gamma)
    crit=gamma*np.sqrt(alpha)+1-1/(1-zeta)*np.sqrt(alpha*(1+3**(4/3)*epsilon**(2/3)*zeta*(1-zeta))/(zeta+(1-zeta)*alpha))
    return crit*(crit>0)

#pacrit hill
def pa_hill(alpha,epsilon,gamma,rC):
    """Classic Hill criterion expressed as a function of p/a (Marchal & Bozis 1982)"""
    zeta=gamma/(1+gamma)
    return (1-zeta)**2*(zeta+(1-zeta)*alpha)/alpha*(gamma*np.sqrt(alpha)+1-rC)**2/(1+3**(4/3)*epsilon**(2/3)*zeta*(1-zeta))


def hill_circular_spacing(epsilon,gamma=1.):
    """
    Return the semi-major axis ratio that corresponds to circular Hill stable planets.
    Exact expression instead of Gladman's criterion
    """
    guess = 1-2.4*epsilon**(1/3)
    return optimize.fsolve(Cc_hill,guess,args=(epsilon,gamma))




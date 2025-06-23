import numpy as np
import pandas as pd
import xarray as xr
import rebound as rb
import reboundx as rbx

pi = np.pi

from scipy.special import poch,factorial2,binom,factorial,gamma,hyp2f1
def lapanddev(s,j,n,alpha):
    """
    Calculates nth derivative with respect to a (alpha) of Laplace coefficient b_s^j(a).
    Uses recursion and scipy special functions. 
    
    Arguments
    ---------
    s : float 
        half-integer parameter of Laplace coefficient. 
    j : int 
        integer parameter of Laplace coefficient. 
    n : int 
        return nth derivative with respect to a of b_s^j(a)
    a : float
        semimajor axis ratio a1/a2 (alpha)
    """    
    #assert alpha>=0 and alpha<1, "alpha not in range [0,1): alpha={}".format(alpha)
    if j<0:
        return lapanddev(s,-j,n,alpha)
    if n >= 2:
        return s * (
            lapanddev(s+1,j-1,n-1,alpha) 
            -  2 * alpha * lapanddev(s+1,j,n-1,alpha)
            + lapanddev(s+1,j+1,n-1,alpha)
            - 2 * (n-1) * lapanddev(s+1,j,n-2,alpha)
        )
    if n==1:
        return s * (
            lapanddev(s+1,j-1,0,alpha) 
            - 2 * alpha * lapanddev(s+1,j,0,alpha) 
            + lapanddev(s+1,j+1,0,alpha)
        )
    return 2 * poch(s,j) * alpha**j * hyp2f1(s,s+j,j+1,alpha**2)/ factorial(j)

def addPoincarevar(ds):
    """ds is as xarray Dataset containing planar orbital elements"""

    ds["Lambda"] = (())


def _Wij(mi,Lj,nj,alij,l,m0=1.):
    return -mi*nj*Lj/2/m0*lapanddev(0.5,l,0,alij)

def _Vijint(mi,Li,Lj,nj,alij,l,m0=1.):
    return mi*nj*Lj/2/m0*np.sqrt(2/Li)*((l+1)*lapanddev(0.5,l,0,alij)+alij/2*lapanddev(0.5,l+1,1,alij))

def _Vijext(mi,Lj,nj,alij,l,m0=1.):
    return -mi*nj*Lj/2/m0*np.sqrt(2/Lj)*((l+1/2)*lapanddev(0.5,l,0,alij)+alij/2*lapanddev(0.5,l,1,alij))

def _dW12L2(m1,n2,al12,l,m0=1.):
    return m1*n2/m0*(lapanddev(0.5,l,0,al12)+al12*lapanddev(0.5,l,1,al12))
def _dW23L2(m3,n2,al23,l,m0=1.):
    return -m3*n2/m0*al23**2*lapanddev(0.5,l,1,al23)

def _dW12L1(m2,n1,al12,l,m0=1.):
    return -m2*n1/m0*al12**2*lapanddev(0.5,l,1,al12)
def _dW23L3(m2,n3,al23,l,m0=1.):
    return m2*n3/m0*(lapanddev(0.5,l,0,al23)+al23*lapanddev(0.5,l,1,al23))

def _dV12intL2(m1,n2,L1,al12,l,m0=1.):
    return -m1*n2/m0*np.sqrt(2/L1)*((l+1)*lapanddev(0.5,l+1,0,al12)+(l+2)*al12*lapanddev(0.5,l+1,1,al12)+al12**2/2*lapanddev(0.5,l+1,2,al12))

def _dV12extL2(m1,n2,L2,al12,l,m0=1.):
    return m1*n2/m0*np.sqrt(2/L2)*((10*l+1)/8*lapanddev(0.5,l,0,al12)+(l+13/8)*al12*lapanddev(0.5,l,1,al12)+al12**2/2*lapanddev(0.5,l,2,al12))

def _dV23intL2(m3,n2,L2,al23,l,m0=1.):
    return m3*n2/m0*np.sqrt(2/L2)*((l+1)/4*lapanddev(0.5,l+1,0,al23)+(l+11/8)*al23*lapanddev(0.5,l+1,1,al23)+al23**2/2*lapanddev(0.5,l+1,2,al23))

def _dV23extL2(m3,n2,L3,al23,l,m0=1.):
    return -m3*n2/m0*al23*np.sqrt(2/L3)*((l+1)*al23*lapanddev(0.5,l,1,al23)+al23**2/2*lapanddev(0.5,l,2,al23))

def _dS12x2(m1,n2,L1,L2,al12,l,m0=1.):
    """
        Returns terms that should be multiplied by bx1 and bx2 respectively
    """
    prefac = -m1*n2/m0
    db_2 = lambda d: al12**d*lapanddev(0.5,l,d,al12)
    f2 = 0.5*(-l**2*db_2(0)+db_2(1)/2+db_2(2)/4)

    db_10 = lambda d: al12**d*lapanddev(0.5,l+1,d,al12)
    f10 = -((-l**2-3/2*l-1/2)*db_10(0)+db_10(1)/2+db_10(2)/4)

    return prefac*np.array([(L2/L1)**.5*f10,f2])

def _dS23x2(m3,n2,L2,L3,al23,l,m0=1.):
    """
        Returns terms that should be multiplied by bx2 and bx3 respectively
    """
    prefac = -m3*n2*al23/m0
    db_2 = lambda d: al23**d*lapanddev(0.5,l,d,al23)
    f2 = 0.5*(-l**2*db_2(0)+db_2(1)/2+db_2(2)/4)

    db_10 = lambda d: al23**d*lapanddev(0.5,l+1,d,al23)
    f10 = -((-l**2-3/2*l-1/2)*db_10(0)+db_10(1)/2+db_10(2)/4)

    return prefac*np.array([f2,(L2/L3)**.5*f10])

def _dU12x2(m1,n2,L1,L2,al12,l,m0=1.):
    """
        Returns terms that should be multiplied by bx1 and bx2 respectively
    """
    prefac = -m1*n2/m0
    db_49 = lambda d: al12**d*lapanddev(0.5,l+1,d,al12)
    f49 = -1/4*((4*l**2+10*l+6)*db_49(0)+(4*l+6)*db_49(1)+db_49(2))

    db_53 = lambda d: al12**d*lapanddev(0.5,l,d,al12)
    f53 = 1/8*((4*l**2+9*l+4)*db_53(0)+(4*l+6)*db_53(1)+db_53(2))

    return prefac*np.array([(L2/L1)**.5*f49,2*f53])

def _dU23x2(m3,n2,L2,L3,al23,l,m0=1.):
    """
        Returns terms that should be multiplied by bx2 and bx3 respectively
    """
    prefac = -m3*n2*al23/m0
    db_49 = lambda d: al23**d*lapanddev(0.5,l+1,d,al23)
    f49 = -1/4*((4*l**2+10*l+6)*db_49(0)+(4*l+6)*db_49(1)+db_49(2))

    db_45 = lambda d: al23**d*lapanddev(0.5,l+2,d,al23)
    f45 = 1/8*((4*l**2+11*l+6)*db_45(0)+(4*l+6)*db_45(1)+db_45(2))

    return prefac*np.array([2*f45,(L2/L3)**.5*f49])




def Rexact(masses, periods,k1,k3, m0=1.,Gr=4*pi**2):
    mu = Gr*m0
    m1,m2,m3 = masses
    ns = 2*pi/periods
    smas = (mu/ns**2)**(1/3)
    Lambdas = masses*np.sqrt(mu*smas)

    al12 = smas[0]/smas[1]
    al23 = smas[1]/smas[2]

    nu12 = periods[0]/periods[1]
    nu23 = periods[1]/periods[2]

    W12 = _Wij(m1,Lambdas[1],ns[1],al12,k1,m0)
    W23 = _Wij(m2,Lambdas[2],ns[2],al23,k3,m0)
    dW12 = _dW12L2(m1,ns[1],al12,k1,m0)
    dW23 = _dW23L2(m3,ns[1],al23,k3,m0)
    V12i = _Vijint(m1,Lambdas[0],Lambdas[1],ns[1],al12,-k1,m0)
    V12e = _Vijext(m1,Lambdas[1],ns[1],al12,-k1,m0)
    V23i = _Vijint(m2,Lambdas[1],Lambdas[2],ns[2],al23,k3-1,m0)
    V23e = _Vijext(m2,Lambdas[2],ns[1],al23,k3-1,m0)
    dV12i = _dV12intL2(m1,ns[1],Lambdas[0],al12,-k1,m0)
    dV12e = _dV12extL2(m1,ns[1],Lambdas[1],al12,-k1,m0)
    dV23i = _dV23intL2(m3,ns[1],Lambdas[1],al23,k3-1,m0)
    dV23e = _dV23extL2(m3,ns[1],Lambdas[2],al23,k3-1,m0)

    V23i_2nd = _Vijint(m2,Lambdas[1],Lambdas[2],ns[2],al23,-k3-1,m0)
    V12e_2nd = _Vijext(m1,Lambdas[1],ns[1],al12,k1,m0)
    dS12 = _dS12x2(m1,ns[1],Lambdas[0],Lambdas[1],al12,k1,m0=1.)
    dS23 = _dS23x2(m3,ns[1],Lambdas[1],Lambdas[2],al23,-k3,m0=1.)
    dU12 = _dU12x2(m1,ns[1],Lambdas[0],Lambdas[1],al12,-k1,m0=1.)
    dU23 = _dU23x2(m3,ns[1],Lambdas[1],Lambdas[2],al23,k3-2,m0=1.)


    R1 = 1/ns[1]*(-W23/(1-nu23)*dV12i+(k1-1)/k3/(1-nu23)*dW23*V12i+3*(k1-1)*W23*V12i/(k3*ns[1]*Lambdas[1]*(1-nu23)**2))

    R1p = 1/ns[1]*(-W23/(1-nu23)*dV12e+(k1-1)/k3/(1-nu23)*dW23*V12e+3*(k1-1)*W23*V12e/(k3*ns[1]*Lambdas[1]*(1-nu23)**2))

    R3 = 1/ns[1]*(W12/(nu12**-1-1)*dV23e-(k3-1)/k1/(nu12**-1-1)*dW12*V23e+3*(k3-1)*W12*V23e/(k1*ns[1]*Lambdas[1]*(nu12**-1-1)**2))

    R3p = 1/ns[1]*(W12/(nu12**-1-1)*dV23i-(k3-1)/k1/(nu12**-1-1)*dW12*V23i+3*(k3-1)*W12*V23i/(k1*ns[1]*Lambdas[1]*(nu12**-1-1)**2))

    print(R1,R1p,R3p,R3,V12e/dU12[1],V12e_2nd/dS12[1])

    #2nd order terms
    V12S23 = -1/(k3*(ns[1]-ns[2]))*V12e*dS23
    V12U23 = 1/(k1*ns[0]-(k1+1)*ns[1])*V12e_2nd*dU23
    V23S12 = 1/(k1*(ns[0]-ns[1]))*V23i*dS12
    V23U12 = -1/((k3+1)*ns[1]-k3*ns[2])*V23i_2nd*dU12

    R1 += (V23S12+V23U12)[0]
    R1p += (V23S12+V23U12)[1]
    R3p += (V12S23+V12U23)[0]
    R3 += (V12S23+V12U23)[1]
    print(R1,R1p,R3p,R3)
    #R = 2**.5*(R1**2/Lambdas[0]+(R1p+R3p)**2/Lambdas[1]+R3**2/Lambdas[2])**.5
    R = (R1**2+(R1p+R3p)**2+R3**2)**.5

    #rys = np.array([R1/R*np.sqrt(2/Lambdas[0]),(R1p+R3p)/R*np.sqrt(2/Lambdas[1]),R3/R*np.sqrt(2/Lambdas[2])])
    rys = np.array([R1/R,(R1p+R3p)/R,R3/R])

    #res = np.array([R1/R,(R1p+R3p)/R,R3/R])

    return R,rys#,res

def epsS0(masses, periods,k1,k3, m0=1.,Gr=4*pi**2):
    mu = Gr*m0
    m1,m2,m3 = masses
    ns = 2*pi/periods
    smas = (mu/ns**2)**(1/3)
    Lambdas = masses*np.sqrt(mu*smas)

    al12 = smas[0]/smas[1]
    al23 = smas[1]/smas[2]

    return k1*(_dW12L1(m2,ns[0],al12,0,m0)+_dW12L1(m3,ns[0],al12*al23,0,m0)) +(1-k1-k3)*(_dW12L2(m1,ns[1],al12,0,m0)+_dW23L2(m3,ns[1],al23,0,m0))+k3*(_dW23L3(m2,ns[2],al23,0,m0)+_dW23L3(m1,ns[2],al23*al12,0,m0))

def Rapprox(masses, periods,k1,k3, m0=1.,Gr=4*pi**2):
    mu = Gr*m0
    m1,m2,m3 = masses
    ns = 2*pi/periods
    smas = (mu/ns**2)**(1/3)
    Lambdas = masses*np.sqrt(mu*smas)

    A = 135/16/pi

    al12 = smas[0]/smas[1]
    al23 = smas[1]/smas[2]

    nu12 = periods[0]/periods[1]
    nu23 = periods[1]/periods[2]

    delta = (1-al12)*(1-al23)/(2-al12-al23)
    nu = (1-nu23)*(1/nu12-1)/(1/nu12-nu23)
    eta = (1-nu23)/(1/nu12-nu23)
    R = A*m1*m3*ns[1]*Lambdas[1]*eta*(1-eta)*abs(k1+k3)/m0**2/delta**2*np.exp(-2*abs(k1+k3)*delta)*np.sqrt(2*eta**2/Lambdas[0]+2/Lambdas[1]+2*(1-eta)**2/Lambdas[2])

    rnorm = np.sqrt(2*eta**2/Lambdas[0]+2/Lambdas[1]+2*(1-eta)**2/Lambdas[2])
    rs = np.array([-eta*np.sqrt(2/Lambdas[0])/rnorm,(2*eta-1)*np.sqrt(2/Lambdas[1])/rnorm,(1-eta)*np.sqrt(2/Lambdas[0])/rnorm])
    return R,rs

def K2exact(masses, periods,k1,k3, m0=1.,Gr=4*pi**2):
    mu = Gr*m0
    m1,m2,m3 = masses
    ns = 2*pi/periods
    smas = (mu/ns**2)**(1/3)
    Lambdas = masses*np.sqrt(mu*smas)
    return 3*(k1**2*ns[0]/Lambdas[0]+(k1+k3-1)**2*ns[1]/Lambdas[1]+k3**2*ns[2]/Lambdas[2])

def epskappa(masses, periods,k1,k3, m0=1.,Gr=4*pi**2):
    R,r = Rexact(masses, periods,k1,k3, m0,Gr)
    K2 = K2exact(masses, periods,k1,k3, m0,Gr)

    return 2**.5*R/K2

def calcLam(masses, periods,m0=1.,Gr=4*pi**2):
    mu = Gr*m0
    m1,m2,m3 = masses
    ns = 2*pi/periods
    smas = (mu/ns**2)**(1/3)
    return masses*np.sqrt(mu*smas)
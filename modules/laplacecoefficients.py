from scipy.special import hyp2f1,gamma,loggamma
import numpy as np

def b_lap(s,l,alpha):
    """
    Laplace coefficient $b_s^{(l)}(alpha)$ computed using the expression provided by Laskar & Robutel 1995
    
    Inputs:
    s : half integer. In this project, we usually have s=1/2 
    l : index of the Fourier coefficient
    alpha : semi-major axis ratio, <1

    Returns:
    Laplace coefficient

    Note that for large indices l, the hypergeometric function evaluation fails.
    The use of loggamma instead of gamma allows to go a bit further
    """
    return 2*np.exp(loggamma(s+l)-loggamma(l+1))/(gamma(s))*alpha**l*hyp2f1(s,s+l,l+1,alpha**2)

def db_lap(s,l,alpha):
    """
    Derivative with respect to alpha of the Laplace coefficients. See b_lap.

    Inputs:
    s : half integer. In this project, we usually have s=1/2 
    l : index of the Fourier coefficient
    alpha : semi-major axis ratio, <1
    """
    return 0.5*(b_lap(s+1,l+1,alpha)+b_lap(s+1,l-1,alpha))-alpha*b_lap(s+1,l,alpha)
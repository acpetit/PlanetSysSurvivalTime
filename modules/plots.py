
import numpy as np
import pandas as pd
import numpy.random as rd
from scipy import stats
import scipy as sc

from math import gcd
from numpy import pi

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import survivaltime as st
import MMRoverlap as overlap
import amdcrit



def zerothMMR_network(pqmax=50,nu=0.05):
    """Plot the zeroth order resonance network.
    Returns the figure
    """

    nu12=np.linspace(0.5,1,100)

    fig,ax = plt.subplots(figsize=(4,4))
    ax.set_aspect('equal', adjustable='box')
    for pq in range(2,pqmax):
        for p in range(1,pq):
            if gcd(p,pq)==1:
                q=pq-p
                mask = ((1-p/q*(1/nu12-1))>0.3)&((1-p/q*(1/nu12-1))<1)
                ax.plot(nu12[mask],(1-p/q*(1/nu12-1))[mask],c=cm.viridis(pq/pqmax),lw=0.5,zorder=2*pqmax-pq)
    for pq in range(1,pqmax):
        if pq < pqmax/4:
            ax.axhline(1-1/(pq+1),c='k',ls='--',lw=0.5,zorder = pqmax*3)
            ax.axvline(1-1/(pq+1),c='k',ls='--',lw=0.5,zorder = pqmax*3)
        if pq < pqmax/8:
            ax.plot(nu12,pq/(pq+1)/nu12,c='k',ls='--',lw=0.5,zorder = pqmax*3)

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=pqmax))
    plt.colorbar(sm,fraction=0.046, pad=0.01,label=r'Resonance index $p+q$')
    ax.set_xlim(0.6,1)
    ax.set_ylim(0.6,1)
    ax.set_yticks(np.linspace(0.6,1,5))

    if nu is not None:
    #Constant nu curve
        nu = 0.05
        nu23 = (1-nu12-nu)/(1-(1+nu)*nu12)

        ax.plot(nu12[nu12<1/(1+nu)],nu23[nu12<1/(1+nu)],'tab:orange',lw=2,zorder=2*pqmax+1)
        text = ax.text(0.63,0.96,r'Constant $\nu$ curve',color='tab:orange',zorder=2*pqmax+1,)
        text.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none'))
    

    ax.set_xlabel(r'Period ratio $\nu_{12} = P_1/P_2$')
    ax.set_ylabel(r'Period ratio $\nu_{23} = P_2/P_3$')
    
    return fig


def number_of_MMR(Nu12,Nu23,Overlap,masses,m0=1.):


    m1,m2,m3 = masses
    nu12min,nu23min = Nu12[0,0],Nu23[0,0]

    fig,ax = plt.subplots(figsize=(4,4))
    ax.set_aspect('equal', adjustable='box')

    boundaries = np.linspace(-.5, 9.5,11)
    cmap_reds = plt.cm.get_cmap('YlOrRd',len(boundaries))
    colors = list(cmap_reds(np.arange(len(boundaries))))
    #replace first color with white
    colors[0] = "white"
    colors[1] = 'grey'
    cmap = matplotlib.colors.ListedColormap(colors[:-1], "")
    # set over-color to last color of list 
    cmap.set_over(colors[-1])

    pcm =plt.pcolormesh(Nu12,Nu23,Overlap,
        cmap=cmap,rasterized=True,
        norm = matplotlib.colors.BoundaryNorm(boundaries, ncolors=len(boundaries)-1, clip=False)
    )
    cb = plt.colorbar(pcm, extend="max",fraction=0.045, pad=0.01)
    cb.set_ticks(np.arange(10))
    cb.set_label('Number of overlapping resonances')
    ax.set_xlim(nu12min,1)
    ax.set_ylim(nu23min,1)

    plt.axhline((amdcrit.hill_circular_spacing((m2+m3)/m0))**1.5,lw=2,c='tab:green')
    plt.axvline((amdcrit.hill_circular_spacing((m1+m2)/m0))**1.5,lw=2,c='tab:green')


    text = ax.text(0.92,0.95,'Hill stability\n limits',color='tab:green',)
    text.set_bbox(dict(facecolor='white', alpha=0.95, edgecolor='none'))

    etas = np.linspace(0,1,1000)
    nuov = overlap.nu_overlaplimit(etas,masses,m0)
    nu12ov,nu23ov = st.nu_eta_to_nus(nuov,etas)

    sl = (nu12ov<1)&(nu23ov<1)
    ax.plot(nu12ov[sl],nu23ov[sl],lw=2)

    text = ax.text(0.73,0.925,'Theoretical overlap limit',color='tab:blue',)
    text.set_bbox(dict(facecolor='white', alpha=0.95, edgecolor='none'))

    ax.set_xlabel(r'Period ratio $\nu_{12} = P_1/P_2$')
    ax.set_ylabel(r'Period ratio $\nu_{23} = P_2/P_3$')

    return fig


def comparison_hill_radius(mm0=np.logspace(-7,-3,100)):
    """Plot the different criteria (fig6) in units of Hill radii
    Inputs mpl/m0
    """
    A=np.sqrt(38/pi)
    Ares = 4*2**.5/3*A
    ### Computation with equal mass and spacing
    coef2pl = 1.46*2**(2/7)
    coef3pl = 2*(1.5)**.5*Ares**.25*(0.5**2)**(3/8)

    ov2pl = coef2pl*mm0**(2/7)
    ov3pl = coef3pl*mm0**.25
    hill = (2/3*mm0)**(1/3) #Normalized hill radius
    hill_exact = 2*(1-amdcrit.hill_circular_spacing(2*mm0))/(1+amdcrit.hill_circular_spacing(2*mm0))

    fig,ax = plt.subplots(figsize=(4,3))
    ax.plot(np.log10(mm0),hill_exact/hill,label="Hill stability",lw=1.5)
    ax.plot(np.log10(mm0),2*ov2pl/(2+ov2pl)/hill,label='2-pl. 1st order MMR overlap',lw=1.5)
    ax.plot(np.log10(mm0),2*ov3pl/(2+ov3pl)/hill,label='3-pl. 0th order MMR overlap',lw=1.5)
    #ax.plot(np.log10(mm0),2*ov3pl_fudge/(2+ov3pl_fudge)/hill,label='3-pl. MMR overlap\n larger width',lw=1.5)

    ax.text(-4.5,6.2,r'$\propto\varepsilon^{-1/12}$',c='tab:green')
    ax.text(-6,4.3,r'$\propto\varepsilon^{-1/21}$',c='tab:orange')

    ax.legend()
    ax.set_xlim(-7,-3)
    ax.set_ylim(2.5,10)

    ax.set_xlabel(r'Planet to star mass ratio $\log_{10}\varepsilon=\log_{10}\frac{m_p}{m_0}$')
    ax.set_ylabel(r'Initial orbital separation in mutual Hill radius')
    return fig

def exittime_distribution(u0s=np.linspace(0.1,0.5,5)):
    fig,ax = plt.subplots(figsize=(4,2))
    times = np.logspace(-4,4,200)
    du=1
    D=1
    for u0 in u0s:
        timefac = u0**2*(1-u0)**2 #Since we sample around the average value (see paper)
        plt.plot(np.log10(times*timefac),
                 times*timefac*np.nan_to_num(st.distribution_exittimes(times*timefac,u0,du,D,N=201)),label = f'$u_0$=${u0:0.02}$',c=cm.viridis_r(1.5*u0))
    plt.ylim(0)
    plt.legend()
    ax.set_xlim(-3.5,1.5)
    ax.set_xlabel(r'$\mathrm{Survival\ time\ }\log_{10}T_{\mathrm{surv}}/T_0$')
    ax.set_ylabel(r'$\mathrm{d}P/\mathrm{d}(\log_{10}T_{\mathrm{surv}}/T_0)$')
    return fig

def mean_and_std_exittimes(u0s=np.linspace(0.01,0.99,1000),fullres=False):
    """
    COmpute and plot the mean and stds of log(exit time)
    
    u0s is the normalized distance to the boundary
    if fullres return means and stds
    """
    fig,axs = plt.subplots(2,figsize=(4,4))

    stds=[]
    means = []
    times = np.logspace(-4,4,1000)
    
    #Create pdf by sampling and then computing mean and std
    for u0 in u0s:
        timefac = u0**2*(1-u0)**2
        exit_timeaspdf=np.maximum(0,np.nan_to_num(st.distribution_exittimes(times*timefac,u0,N=201)))*np.diff(times*timefac,prepend=0)
        exit_timeaspdf = exit_timeaspdf/np.sum(exit_timeaspdf)
        exitimes = rd.choice(times*timefac,500000,p=exit_timeaspdf)
        stds.append(np.log10(exitimes).std())
        means.append(np.log10(exitimes).mean())
    stds=np.array(stds)
    means=np.array(means)

    axs[0].plot(u0s,means,label=r'$\mathrm{Mean\ }\langle\log_{10} T_{\mathrm{surv}}/T_0\rangle$',ls='--',zorder=2)
    axs[0].plot(u0s,2*np.log10(u0s*(1-u0s))+np.log10(1.5),label=r'$\log_{10}\left(\frac{3}{2}u_0^2(1-u_0)^2\right)$',zorder=1)
    axs[0].legend()
    axs[0].set_xlim(0,1)
    axs[1].set_xlabel(r'$u_0$')
    axs[0].set_ylabel(r'$\langle\log_{10} T_{\mathrm{surv}}/T_0\rangle\mathrm{\ (dex)}$')

    axs[1].plot(u0s,stds,label=r'$\sigma\left(\log_{10} T_{\mathrm{surv}}/T_0\right)$',ls='--',zorder=2)
    axs[1].plot(u0s,0.9-2.25*u0s*(1-u0s),label=r'$0.9-2.25u_0(1-u_0)$',zorder=1)
    axs[1].set_xlim(0,1)
    axs[1].set_ylabel(r'$\mathrm{Standard\ deviation\ (dex)}$')
    axs[1].legend()

    axs[1].fill_between(np.linspace(0,1),0.43-0.16,0.43+0.16,color='tab:green',alpha=0.1)
    axs[1].axhline(0.43,c='tab:green',ls='--')
    axs[1].text(0.02,0.36,r'$0.43\pm0.16$',c='tab:green')
    if fullres:
        return fig,means,stds
    else:
        return fig



def EMS_survtime(df_times,mass=1e-5,range_perrat=(1.0,1.5),real_overlap=True):
    fig,ax = plt.subplots(figsize=(4,3))

    plt.semilogy(df_times.per_rat,df_times.time,'.',c='tab:blue',ms=1.5,label='3 Planets')
    plt.semilogy(df_times.per_rat,10**np.log10(df_times.Tsurv),c='tab:red',label='3 Planets',lw=1.7)
    for k in range(1,70):
        plt.axvline(1+1/k,ls='--',lw=0.7,c='k')
        if not k%2:
            plt.axvline((k+1)/(k-1),ls='--',lw=0.7,c='tab:green')
        plt.fill_betweenx([1,1e10],
                    (1+1/k)*(1-4.18/2*1.5*(k)**(1/3)*(2*mass)**(2/3)),(1+1/k)*(1+4.18/2*1.5*(k)**(1/3)*(2*mass)**(2/3)),
                        alpha=0.1,color='tab:orange')#Resonance width from Petit et al. 2017
    plt.axvline((amdcrit.hill_circular_spacing(2*mass,1))**(-3/2),lw=2)
    plt.axvline((1-1.46*(2*mass)**(2/7))**(-3/2),lw=2,c='tab:orange')
    if real_overlap:
        plt.axvline(np.exp(3*st.get_plsep_ov(0.8,0.8,mass*np.ones(3))),c='tab:red',ls='--',lw=2)#eq. 84
    plt.ylim(2,2e9)
    plt.xlim(*range_perrat)
    plt.xlabel(r'Initial period ratio $P_{j+1}/P_j$')
    plt.ylabel('Survival time (in inner orbit period)')
    return fig
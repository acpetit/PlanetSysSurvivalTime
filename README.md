# The path to instability in multi-planetary systems

This repository contains the source code to reproduce the figures from Petit et al. (2020). The figure notebooks can be run directly from the csv files in the repo. The notebook with the figures is instability_multi.ipynb.

There are also python modules containing the function used to compute the survival time of any system, the main function being in survivaltime.py. The amdcrit module contains functions allowing to compute the critical AMD for a pair of planet.


## Known typos in the paper

- In eq. 32, the Laplace coefficient should read $b_{1/2}^{(l+1)}(\alpha_{ij})$ (corrected in Petit 2021 eq. 15, noted by J. Couturier).
- Eq. 61 that describes the overlap criterion in the special case equal masses and spacing should read: $\delta_{\mathrm{ov,eq}} = 1.0\left(\frac{m_p}{m_0}\right)^{1/4}$ (noted by D. Tamayo and S. Hadden).
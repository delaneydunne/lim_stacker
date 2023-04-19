import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import h5py
from astropy.io import fits
import csv

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord

# standard COMAP cosmology
cosmo = FlatLambdaCDM(H0=70*u.km / (u.Mpc*u.s), Om0=0.286, Ob0=0.047)

# funky plot packages
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import SymLogNorm

import lim_stacker as st

""" INPUT FILES """
# path to COMAP map files
mapfiles = glob.glob('yr1data/*_summer.h5')
mapfiles = [mapfiles[0], mapfiles[2], mapfiles[1]]

 # path to the galaxy catalogue (I made preliminary cuts before running it through)
galcatfile = 'BOSS_quasars/cutquasarcat.npz'

""" PARAMETERS """
# set up a params class that you can just pass around
# if you'd like to use non-default values, pass a file to st.parameters()
params = st.parameters()

""" SETUP """
comaplist, qsolist = st.setup(mapfiles, galcatfile, params)

""" RUN """
returns = st.stacker(comaplist, qsolist, params)
# for a list of returns, see help(st.stacker)

# dict of output statistics
stackvals = returns[0]

print("stack line luminosity is {:.3e} +/- {:.3e} K km/s pc^2".format(stackvals['linelum'], stackvals['dlinelum']))

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import h5py
from astropy.io import fits

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord

# funky plot packages
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import SymLogNorm

# beam aperture extraction
# from photutils.aperture import CircularAperture, aperture_photometry

import lim_stacker as st

"""example file to run the code"""

# load in the data
mapfiles = glob.glob('data/*_summer.h5')
mapfiles = [mapfiles[0], mapfiles[2], mapfiles[1]]
galcatfile = 'data/cutquasarcat.npz'

""" PARAMETERS """
# set up a params class that you can just pass around
params = st.empty_table()
params.xwidth = 3 # number of x pixels to average between when getting the cutout T
params.ywidth = 3 # number of y pixels to average between when getting the cutout T
params.freqwidth = 2 # number of freq pixels to average between when getting the cutout T

params.centfreq = 115.27 # rest frequency CO(1-0)
params.beamwidth = 1 # when smoothing to the synthesized beam, std of gaussian kernel
params.gauss_kernel = None # Gaussian2DKernel(params.beamwidth)
params.tophat_kernel = None # Tophat2DKernel(params.beamwidth)
params.spacestackwidth = None # in pixels -- if you only want single T value from each cutout, set to None
params.freqstackwidth = None # number of channels. "" ""
params.obsunits = False
params.verbose = False
# plotting parameters
params.savepath = 'output'
params.saveplots = False
params.plotspace = False
params.plotfreq = False
params.fieldcents = [SkyCoord(25.435*u.deg, 0.0*u.deg), SkyCoord(170.0*u.deg, 52.5*u.deg),
                     SkyCoord(226.0*u.deg, 55.0*u.deg)]
params.cubelet = False
# bootstrap parameters
params.nzbins = 3
# ***save as you go
params.itersave = True
params.itersavefile = "randomstacker_output.npz"
params.itersavestep = 10

""" SET UP THE DATA """
comaplist, qsolist = st.setup(mapfiles, galcatfile, params)

""" DO A REAL STACK TO GET THE REAL DISTRIBUTION """
tidxlist = st.random_stacker_setup(comaplist, qsolist, params)

print('Done Setup')

""" STACKS ON RANDOM LOCATIONS """
# ****fix your VERY bad saving iteratively!!
""" STACKS ON RANDOM LOCATIONS """
Tvals, Trmsvals = st.n_random_stacks(10000, tidxlist, comaplist, qsolist, params,
                                         verbose=True, itersave=True)

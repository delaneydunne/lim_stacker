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

# set up a params class that you can just pass around
params = st.empty_table()
params.xwidth = 2 # number of x pixels to average between when getting the cutout T
params.ywidth = 2 # number of y pixels to average between when getting the cutout T
params.freqwidth = 2 # number of freq pixels to average between when getting the cutout T

# cent vals (for properly centering the cutout)
length = params.xwidth // 2
params.idxmin, params.idxmax = length-1, length+1

params.centfreq = 115.27 # rest frequency CO(1-0)
params.beamwidth = 1 # when smoothing to the synthesized beam, std of gaussian kernel
params.gauss_kernel = None # Gaussian2DKernel(params.beamwidth)
params.tophat_kernel = None # Tophat2DKernel(params.beamwidth)
params.spacestackwidth = None # in pixels -- if you only want single T value from each cutout, set to None
params.freqstackwidth = None # number of channels. "" ""

# plotting parameters
params.savepath = 'beamscale_output'
params.saveplots = False
params.plotspace = False
params.plotfreq = False
params.fieldcents = [SkyCoord(25.435*u.deg, 0.0*u.deg), SkyCoord(170.0*u.deg, 52.5*u.deg),
                     SkyCoord(226.0*u.deg, 55.0*u.deg)]

params.beamscale=False
beamscale = np.array([[0.25, 0.5, 0.25],
                      [0.50, 1.0, 0.50],
                      [0.25, 0.5, 0.25]])
beamscale3d = np.tile(beamscale, (params.freqwidth, 1, 1))

params.beam = beamscale3d

params.nzbins = 3

comaplist, qsolist = st.setup(mapfiles, galcatfile, params)

tidxlist = st.random_stacker_setup(comaplist, qsolist, params)

Tvals, Trmsvals = st.n_random_stacks(10000, tidxlist, comaplist, qsolist, params, verbose=True)
np.savez('randomstacker_values.npz', T=Tvals, rms=Trmsvals)

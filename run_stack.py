import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import h5py
from astropy.io import fits

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70*u.km / (u.Mpc*u.s), Om0=0.286, Ob0=0.047)
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord

# funky plot packages
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import SymLogNorm

import stacker as st

# path to COMAP map files
mapfiles = glob.glob('yr1data/*_summer.h5')
mapfiles = [mapfiles[0], mapfiles[2], mapfiles[1]]

 # path to the galaxy catalogue (I made preliminary cuts before running it through)
galcatfile = 'BOSS_quasars/cutquasarcat.npz'

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
params.gauss_kernel = Gaussian2DKernel(params.beamwidth)
params.tophat_kernel = Tophat2DKernel(params.beamwidth)
params.spacestackwidth = 10 # in pixels -- if you only want single T value from each cutout, set to None
params.freqstackwidth = 20 # number of channels. "" ""

# plotting parameters
params.savepath = 'output'
params.saveplots = True
params.plotspace = True
params.plotfreq = True
params.fieldcents = [SkyCoord(25.435*u.deg, 0.0*u.deg), SkyCoord(170.0*u.deg, 52.5*u.deg),
                     SkyCoord(226.0*u.deg, 55.0*u.deg)]

params.beamscale=False
beamscale = np.array([[0.25, 0.5, 0.25],
                      [0.50, 1.0, 0.50],
                      [0.25, 0.5, 0.25]])
beamscale3d = np.tile(beamscale, (params.freqwidth, 1, 1))

params.beam = beamscale3d

comaplist, qsolist = st.setup(mapfiles, galcatfile, params)

stacktemp, stackrms, image, spectrum, qsoidxlist = st.stacker(comaplist, qsolist, params)

print("stack Tb is {:.3e} +/- {:.3e} uK".format(stacktemp*1e6, stackrms*1e6))

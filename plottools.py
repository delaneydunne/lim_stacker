from __future__ import print_function
from .tools import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import matplotlib.gridspec as gridspec
from matplotlib import get_cmap
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometr
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
import os
import h5py
import csv
import warnings
import copy
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", message="invalid value encountered in power")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")

cmap = get_cmap('twilight')

""" CUBELET PLOTS """

def changrid(cubelet, params, smooth=None, rad=None, ext=None, offset=0):
    """
    plots spatial images of the 4 adjacent channels on either side of the central
    channel of the cubelet. if offset != 0, offsets the central channel by offset.
    if rad is set, will only include rad spaxels on either side of the central
    spaxel. ext is the % of maximum used to set vmin/vmax, which will be centered
    around zero always
    """

    fig = plt.figure(figsize=(9,7))
    supgs = gridspec.GridSpec(1,1,figure=fig)
    gs = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=supgs[0])
    axs = gs.subplots(sharex='col', sharey='row')

    if not ext:
        if smooth:
            ext = 0.25
        else:
            ext = 0.7

    vmin, vmax = np.nanmin(cubelet)*ext, np.nanmax(cubelet)*ext

    freqcent = int(cubelet.shape[0] / 2) + offset
    spaceext = cubelet.shape[1]
    spacecent = int(spaceext / 2)

    if rad:
        xyext = [spacecent - rad, spacecent + rad]
        spacecent = rad
    else:
        xyext = [0,spaceext]



    rectmin = spacecent - 1
    rectmax = spacecent + 2

    xcorners = (rectmin, rectmin, rectmax, rectmax, rectmin)
    ycorners = (rectmin, rectmax, rectmax, rectmin, rectmin)

    for i in range(3):
        for j in range(3):

            chan = freqcent - 4 + i*3 + j

            if smooth:
                plotim = convolve(cubelet[chan,xyext[0]:xyext[1],xyext[0]:xyext[1]], params.gauss_kernel)
            else:
                plotim = cubelet[chan,xyext[0]:xyext[1],xyext[0]:xyext[1]]

            c = axs[i,j].pcolormesh(plotim, cmap='PiYG_r', vmin=vmin, vmax=vmax)
            axs[i,j].set_title('channel '+str(chan), fontsize='small')
            axs[i,j].set_aspect(aspect=1)
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])

            axs[i,j].plot(xcorners, ycorners, color='k')

    fig.colorbar(c, ax=axs)

    return fig

def radprof(cubelet, params, chan=None):
    """
    gets the integrated Tb in circular annuli extending radially outwards from
    the central spaxel in a given channel
    if chan isn't set, will default to the central frequency channel in the
    cubelet
    """

    # indexing
    freqcent = int(cubelet.shape[0] / 2)
    spaceext = cubelet.shape[1]
    spacecent = int(spaceext / 2)

    if not chan:
        chan = freqcent

    # central circular aperture
    cent = CircularAperture((spacecent, spacecent), 0.5)
    centmask = cent.to_mask()

    # the rest of the annuli
    aplist = [cent]
    for r in np.arange(spacecent - 1) + 1.5:
        aper = CircularAnnulus((spacecent, spacecent), r-1, r)
        aplist.append(aper)

    proftable = aperture_photometry(cubelet[chan,:,:], aplist)

    sumlist = []
    for i in np.arange(len(aplist)):
        val = float(proftable['aperture_sum_'+str(i)])
        sumlist.append(val)

    sumarr = np.array(sumlist)

    return sumarr

def radprofoverplot(cubelet, rmslet, params, nextra=3, offset=0):
    """
    plots the radial profile of the central frequency channel (offset by offset
    if nonzero), and nextra channels on either side of the central one. will
    also shade in the 1-sigma rms around the central channel
    """

    # indexing
    nchans = nextra*2 + 1
    freqcent = int(cubelet.shape[0] / 2) + offset
    chans = np.arange(nchans) + freqcent - nextra

    fig, ax = plt.subplots(1, figsize=(7, 5))

    carr = np.arange(len(chans)+3) / (len(chans) + 3) + 0.2

    chanprofs = []
    for i, chan in enumerate(chans):
        chanprof = radprof(cubelet, params, chan=chan)
        chanprofs.append(chanprof)

        if chan == freqcent:
            ax.step(np.arange(len(chanprof))*2, chanprof*1e6, zorder=20, where='mid',
                    color=cmap(carr[i]), lw=3, label='Channel {}'.format(str(chan)))

            rmsprof = radprof(rmslet, params, chan=chan)
            print(rmsprof)
            ax.fill_between(np.arange(len(chanprof))*2, (chanprof-rmsprof)*1e6, (chanprof+rmsprof)*1e6,
                            color='0.9', zorder=0)
        else:

            ax.step(np.arange(len(chanprof))*2, chanprof*1e6, zorder=10, where='mid',
                    color=cmap(carr[i]), label='Channel {}'.format(str(chan)))

    ax.axhline(0, color='k', ls='--')
    ax.axvline(0, color='k', ls='--')

    ax.axvline(3, color='0.3', ls=':', zorder=5)

    chanprofs = np.stack(chanprofs)

    axext = np.max(np.abs((np.nanmax(chanprofs), np.nanmin(chanprofs)))) * 1.05 * 1e6
    ax.set_ylim((-axext, axext))

    ax.set_ylabel(r'$T_b$ ($\mu$K)')

    ax.legend(loc='upper left')

    ax.set_xlabel('Radius (arcmin)')

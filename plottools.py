from __future__ import print_function
from .tools import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import matplotlib.gridspec as gridspec
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve, Gaussian2DKernel, Box2DKernel
import os
import h5py
import csv
import warnings
import copy
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", message="invalid value encountered in power")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")

cmap = plt.get_cmap('twilight')




""" MAP PLOTTING FUNCTIONS """
def plot_mom0(comap, params, ext=0.95, lognorm=True, smooth=False):

    """
    unsure about the transpose thing
    """

    fig,ax = plt.subplots(1)

    moment0 = weightmean(comap.map, comap.rms, axis=(0))[0] * 1e6
    if smooth:
        moment0 = convolve(moment0, params.gauss_kernel)

    vext = (np.nanmin(moment0)*ext, np.nanmax(moment0)*ext)

    if lognorm:
        c = ax.pcolormesh(comap.ra, comap.dec, moment0,
                          norm=SymLogNorm(linthresh=1, linscale=0.5,
                                          vmin=vext[0], vmax=vext[1]),
                          cmap='PiYG_r')
    else:
        c = ax.pcolormesh(comap.ra, comap.dec, moment0.T, cmap='PiYG_r')
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')

    cbar = fig.colorbar(c, ax=ax, extend='both')
    cbar.ax.set_ylabel(r'$T_b \ (\mu K)$')

    return fig

def plot_chan(comap, channel, params, ext=0.95, smooth=False, lognorm=True):

    fig,ax = plt.subplots(1)

    plotmap = comap.map[channel,:,:] * 1e6
    if smooth:
        plotmap = convolve(plotmap, params.gauss_kernel)

    vext = (np.nanmin(plotmap)*ext, np.nanmax(plotmap)*ext)
    if lognorm:
        c = ax.pcolormesh(comap.ra, comap.dec, plotmap,
                          norm=SymLogNorm(linthresh=1, linscale=0.5,
                                          vmin=vext[0], vmax=vext[1]),
                          cmap='PiYG_r')
    else:
        c = ax.pcolormesh(comap.ra, comap.dec, plotmap, cmap='PiYG_r')
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')

    cbar = fig.colorbar(c, ax=ax, extend='both')
    cbar.ax.set_ylabel(r'$T_b \ (\mu K)$')

    return fig

""" SINGLE-CUTOUT PLOTS """
def display_cutout(cutout, comap, params, save=None, ext=1.):

    cutoutra = comap.ra[cutout.spacexidx[0]:cutout.spacexidx[1]+1]
    cutoutdec = comap.dec[cutout.spaceyidx[0]:cutout.spaceyidx[1]+1]

    beamra = comap.ra[cutout.xidx[0]:cutout.xidx[1]+1]
    beamdec = comap.dec[cutout.yidx[0]:cutout.yidx[1]+1]

    beamxidx = cutout.xidx - cutout.spacexidx[0]
    beamyidx = cutout.yidx - cutout.spaceyidx[0]

    beamcut = cutout.spacestack[beamxidx[0]:beamxidx[1]+1, beamyidx[0]:beamyidx[1]+1]

    fig,ax = plt.subplots(1)
    vext = np.max(np.abs((np.max(cutout.spacestack), np.min(cutout.spacestack)))) * ext
    c = ax.pcolormesh(cutoutra, cutoutdec, cutout.spacestack, cmap='PiYG_r', vmin=-vext, vmax=vext)
    ax.pcolormesh(beamra, beamdec, beamcut, cmap='PiYG_r', vmin=-vext, vmax=vext, ec='k')

    ax.scatter(cutout.x, cutout.y, color='k', s=2, label="Cutout Tb = {:.2e}".format(cutout.T))
    ax.scatter(cutout.x, cutout.y, color='k', s=2, label="Cutout RMS = {:.2e}".format(cutout.rms))
    ax.scatter(cutout.x, cutout.y, color='k', s=2, label="Catalogue frequency = {:.4f}".format(cutout.freq))

    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')

    ax.legend(loc='upper left')
    fig.colorbar(c)

    if save:
        plt.savefig(save)

    return fig, ax

""" CUBELET PLOTS """
def changrid(cubelet, params, smooth=None, rad=None, ext=None, offset=0):

    fig = plt.figure(figsize=(9,7))
    supgs = gridspec.GridSpec(1,1,figure=fig)
    gs = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=supgs[0])
    axs = gs.subplots(sharex='col', sharey='row')

    if not ext:
        if smooth:
            ext = 0.25
        else:
            ext = 0.7

    vext = np.max(np.abs((np.nanmin(cubelet), np.nanmax(cubelet))))
    vmin, vmax = -vext*ext, vext*ext

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

    return fig

def spaceweightmean(cubelet, rmslet):

    ccubelet = np.ones_like(cubelet)
    crmslet = np.ones_like(cubelet)

    padcubelet = np.pad(cubelet, ((0,0), (1,1), (1,1)), 'constant', constant_values=np.nan)
    padrmslet = np.pad(rmslet, ((0,0), (1,1), (1,1)), 'constant', constant_values=np.nan)

    for i in range(cubelet.shape[1]):
        for j in range(cubelet.shape[2]):

            ii1, ii2 = i, i+3
            ij1, ij2 = j, j+3

            t, rms = weightmean(padcubelet[:,ii1:ii2,ij1:ij2], padrmslet[:,ii1:ii2,ij1:ij2], axis=(1,2))
            ccubelet[:,i,j] = t
            crmslet[:,i,j] = rms

    return ccubelet, crmslet

def specgridx(cubelet, rmslet, nextra=3, offset=0):

    xkernel = Box2DKernel(3)
    xcent = int(cubelet.shape[1] / 2)
    nspax = nextra*2 + 1
    spax = np.arange(nspax) + xcent - nextra

    freqcent = int(cubelet.shape[0] / 2)

    convcube, convrms = spaceweightmean(cubelet, rmslet)

    xarr = convcube[:,:,xcent]
    xrms = convrms[:,:,xcent]

    fig, axs = plt.subplots(nspax, sharex=True, figsize=(5, nspax*1.5))

    vext = np.max([np.abs(np.nanmin(xarr[:,spax])), np.abs(np.nanmax(xarr[:,spax]))])
    vext = np.array((-vext, vext)) * 1e6 * 1.05

    for i, spix in enumerate(spax):
        axs[i].step(np.arange(len(xarr)), xarr[:,spix]*1e6, color='indigo', zorder=10, where='mid',
                    label='{}'.format(str(spix)))
        axs[i].bar(np.arange(len(xarr)), xrms[:,spix]*1e6, width=1, color='0.8', zorder=0, alpha=0.5)
        axs[i].bar(np.arange(len(xarr)), -xrms[:,spix]*1e6, width=1, color='0.8', zorder=0, alpha=0.5)


        axs[i].axhline(0, color='k', ls='--')

        apmin, apmax = freqcent - 0.5, freqcent + 0.5

        axs[i].fill_betweenx(vext, np.ones(2)*apmin, np.ones(2)*apmax, color='0.5', zorder=1, alpha=0.5)
        axs[i].axvline(apmin, color='0.5', ls=':')
        axs[i].axvline(apmax, color='0.5', ls=':')

        axs[i].legend(loc='lower right')
        axs[i].set_ylabel(r'$T_b$ ($\mu$K)')
        axs[i].set_ylim(vext)

    axs[-1].set_xlabel('Channel')

    return fig

def specgridy(cubelet, rmslet, nextra=3, offset=0):

    xkernel = Box2DKernel(3)
    ycent = int(cubelet.shape[1] / 2)
    nspax = nextra*2 + 1
    spax = np.arange(nspax) + ycent - nextra

    freqcent = int(cubelet.shape[0] / 2)


    convcube, convrms = spaceweightmean(cubelet, rmslet)


    yarr = convcube[:,ycent,:]
    yrms = convrms[:,ycent,:]

    fig, axs = plt.subplots(nspax, sharex=True, figsize=(5, nspax*1.5))

    vext = np.max([np.abs(np.nanmin(yarr[:,spax])), np.abs(np.nanmax(yarr[:,spax]))])
    vext = np.array((-vext, vext)) * 1e6 * 1.05

    for i, spix in enumerate(spax):
        axs[i].step(np.arange(len(yarr)), yarr[:,spix]*1e6, color='indigo', zorder=10, where='mid',
                    label='{}'.format(str(spix)))

        axs[i].bar(np.arange(len(yarr)), yrms[:,spix]*1e6, width=1, color='0.8', zorder=0, alpha=0.5)
        axs[i].bar(np.arange(len(yarr)), -yrms[:,spix]*1e6, width=1, color='0.8', zorder=0, alpha=0.5)

        axs[i].axhline(0, color='k', ls='--')

        apmin, apmax = freqcent-0.5, freqcent+0.5

        axs[i].fill_betweenx(vext, np.ones(2)*apmin, np.ones(2)*apmax, color='0.5', zorder=0, alpha=0.5)
        axs[i].axvline(apmin, color='0.5', ls=':')
        axs[i].axvline(apmax, color='0.5', ls=':')

        axs[i].legend(loc='lower right')
        axs[i].set_ylabel(r'$T_b$ ($\mu$K)')
        axs[i].set_ylim(vext)

    axs[-1].set_xlabel('Channel')

    return fig

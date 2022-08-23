from __future__ import print_function
from .tools import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
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

def plot_chan(comap, channel, params, cat=None, ext=0.95, smooth=False, lognorm=True):
    """
    plot a single channel of the input intensity map
    if cat is passed, will also scatter plot objects in the catalogue in that channel
    """

    fig,ax = plt.subplots(1)

    # plot the map
    plotmap = comap.map[channel,:,:] * 1e6
    # convovle with the beam if smooth=True
    if smooth:
        plotmap = convolve(plotmap, params.gauss_kernel)

    # limits of the colourmap (ext is the passed factor by which they're multiplied)
    vext = (np.nanmin(plotmap)*ext, np.nanmax(plotmap)*ext)
    # log colourscale
    if lognorm:
        c = ax.pcolormesh(comap.ra, comap.dec, plotmap,
                          norm=SymLogNorm(linthresh=1, linscale=0.5,
                                          vmin=vext[0], vmax=vext[1]),
                          cmap='PiYG_r')
    # normal colourscale
    else:
        c = ax.pcolormesh(comap.ra, comap.dec, plotmap, cmap='PiYG_r')

    # if a catalogue is passed, scatter plot
    if cat:
        # catalogue objects in the given channel
        chancat = cat.cull_to_chan(comap, params, channel, in_place=False)
        # scatter plot them
        ax.scatter(chancat.ra(), chancat.dec(), color='k', s=2)


    # labels and stuff
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')

    cbar = fig.colorbar(c, ax=ax, extend='both')
    cbar.ax.set_ylabel(r'$T_b \ (\mu K)$')

    return fig

""" SINGLE-CUTOUT PLOTS """
def display_cutout(cutout, comap, params, save=None, ext=1.0):

    cutoutra = comap.ra[cutout.spacexidx[0]:cutout.spacexidx[1]+1]
    cutoutdec = comap.dec[cutout.spaceyidx[0]:cutout.spaceyidx[1]+1]

    beamra = comap.ra[cutout.xidx[0]:cutout.xidx[1]+1]
    beamdec = comap.dec[cutout.yidx[0]:cutout.yidx[1]+1]

    beamxidx = cutout.xidx - cutout.spacexidx[0]
    beamyidx = cutout.yidx - cutout.spaceyidx[0]

    try:
        cutim = cutout.spacestack * 1e6
    except:
        # only want the beam for the axes that aren't being shown
        lcfidx = (cutout.cubestack.shape[0] - params.freqwidth) // 2
        cfidx = (lcfidx, lcfidx + params.freqwidth)

        cutim, imrms = weightmean(cutout.cubestack[cfidx[0]:cfidx[1],:,:],
                                     cutout.cubestackrms[cfidx[0]:cfidx[1],:,:], axis=0)
        cutim *= 1e6
        imrms *= 1e6

    beamcut = cutim[beamxidx[0]:beamxidx[1], beamyidx[0]:beamyidx[1]]

    fig,ax = plt.subplots(1)

    vext = np.max(np.abs((np.nanmax(cutim), np.nanmin(cutim)))) * ext

    c = ax.pcolormesh(cutoutra, cutoutdec, cutim, cmap='PiYG_r', vmin=-vext, vmax=vext)
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

    return fig

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

""" WRAPPER FOR CUBELET PLOTS """
def cubelet_plotter(cubelet, rmslet, params):
    """
    wrapper function to make extra diagnostic plots of the output cubelet from a stack
    """

    # if saving plots, set up the directory structure
    if params.saveplots:
        outputdir = params.savepath + '_x'+str(params.xwidth)+'f'+str(params.freqwidth)

        params.cubesavepath = outputdir + '/plots/cubelet'

        if not os.path.exists(params.cubesavepath):
            os.makedirs(params.cubesavepath)

    # plot a 9x9 grid of spatial images of adjacent channels to the central one
    changridfig = changrid(cubelet, params)
    if params.saveplots:
        changridfig.savefig(params.cubesavepath+'/channel_grid_unsmoothed.png')
    # smoothed version of this
    smchangridfig = changrid(cubelet, params, smooth=True)
    if params.saveplots:
        changridfig.savefig(params.cubesavepath+'/channel_grid_smoothed.png')

    # radial profile of the stack in the central and adjacent channels
    radproffig = radprofoverplot(cubelet, rmslet, params)
    if params.saveplots:
        radproffig.savefig(params.cubesavepath+'/radial_profiles.png')

    # 9x1 grid of spectra in adjacent spatial pixels to center of stack
    #  x-axis
    xspecfig = specgridx(cubelet, rmslet)
    if params.saveplots:
        xspecfig.savefig(params.cubesavepath+'/spectral_profiles_x.png')

    #  y-axis
    yspecfig = specgridy(cubelet, rmslet)
    if params.saveplots:
        yspecfig.savefig(params.cubesavepath+'/spectral_profiles_y.png')

    return


""" STACK OUTPUT PLOTS """
def spatial_plotter(stackim, params, cmap='PiYG_r'):

    # corners for the beam rectangle
    if params.xwidth % 2 == 0:
        rectmin = params.spacestackwidth - params.xwidth/2
        rectmax = params.spacestackwidth + params.xwidth/2
    else:
        rectmin = params.spacestackwidth - params.xwidth // 2
        rectmax = params.spacestackwidth + params.xwidth // 2 + 1

    xcorners = (rectmin, rectmin, rectmax, rectmax, rectmin)
    ycorners = (rectmin, rectmax, rectmax, rectmin, rectmin)

    vext = np.nanmax(np.abs([np.nanmin(stackim*1e6), np.nanmax(stackim*1e6)]))
    vmin,vmax = -vext, vext

    # unsmoothed
    fig, ax = plt.subplots(1)
    c = ax.pcolormesh(stackim*1e6, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.plot(xcorners, ycorners, color='0.8', linewidth=4, zorder=10)
    cbar = fig.colorbar(c)
    cbar.ax.set_ylabel('Tb (uK)')

    if params.saveplots:
        fig.savefig(params.plotsavepath+'/angularstack_unsmoothed.png')

    # smoothed
    smoothed_spacestack_gauss = convolve(stackim, params.gauss_kernel)

    vext = np.nanmax(smoothed_spacestack_gauss*1e6)

    fig, ax = plt.subplots(1)
    c = ax.pcolormesh(smoothed_spacestack_gauss*1e6, cmap=cmap, vmin=-vext, vmax=vext)
    ax.plot(xcorners, ycorners, color='0.8', linewidth=4, zorder=10)
    cbar = fig.colorbar(c)
    cbar.ax.set_ylabel('Tb (uK)')
    ax.set_aspect(aspect=1)

    if params.saveplots:
        fig.savefig(params.plotsavepath+'/angularstack_smoothed.png')

    return 0

def spectral_plotter(stackspec, params):
    fig, ax = plt.subplots(1)
    if params.freqwidth % 2 == 0:
        freqarr = np.arange(params.freqstackwidth * 2)*31.25e-3 - (params.freqstackwidth-0.5)*31.25e-3
    else:
        freqarr = np.arange(params.freqstackwidth * 2 + 1)*31.25e-3 - (params.freqstackwidth)*31.25e-3

    ax.step(freqarr, stackspec*1e6,
            color='indigo', zorder=10, where='mid')
    ax.set_xlabel(r'$\Delta_\nu$ [GHz]')
    ax.set_ylabel(r'T$_b$ [$\mu$K]')
    ax.set_title('Stacked over {} Spatial Pixels'.format((params.xwidth)**2))

    ax.axhline(0, color='k', ls='--')
    ax.axvline(0, color='k', ls='--')

    # show which channels contribute to the stack
    ax.axvline(0 - params.freqwidth / 2 * 31.25e-3, color='0.7', ls=':')
    ax.axvline(0 + params.freqwidth / 2 * 31.25e-3, color='0.7', ls=':')

    if params.saveplots:
        fig.savefig(params.plotsavepath + '/frequencystack.png')

    return 0

def combined_plotter(stackim, stackspec, params, cmap='PiYG_r', stackresult=None):

    # corners for the beam rectangle
    if params.xwidth % 2 == 0:
        rectmin = params.spacestackwidth - params.xwidth/2
        rectmax = params.spacestackwidth + params.xwidth/2
    else:
        rectmin = params.spacestackwidth - params.xwidth // 2
        rectmax = params.spacestackwidth + params.xwidth // 2 + 1

    xcorners = (rectmin, rectmin, rectmax, rectmax, rectmin)
    ycorners = (rectmin, rectmax, rectmax, rectmin, rectmin)

    vext = np.nanmax(np.abs([np.nanmin(stackim*1e6), np.nanmax(stackim*1e6)]))
    vmin,vmax = -vext, vext

    # plot with all three stack representations
    gs_kw = dict(width_ratios=[1,1], height_ratios=[3,2])
    fig,axs = plt.subplots(2,2, figsize=(7,5), gridspec_kw=gs_kw)
    gs = axs[0,-1].get_gridspec()

    for ax in axs[1,:]:
        ax.remove()

    freqax = fig.add_subplot(gs[-1,:])

    c = axs[0,0].pcolormesh(stackim*1e6, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0,0].plot(xcorners, ycorners, color='0.8', linewidth=4, zorder=10)
    axs[0,0].set_title('Unsmoothed')

    axs[0,0].tick_params(axis='y',
                            labelleft=False,
                            labelright=False,
                            left=False,
                            right=False)
    axs[0,0].tick_params(axis='x',
                         labeltop=False,
                         labelbottom=False,
                         top=False,
                         bottom=False)

    divider = make_axes_locatable(axs[0,0])
    cax0 = divider.new_horizontal(size='5%', pad=0.05)
    fig.add_axes(cax0)
    cbar = fig.colorbar(c, cax=cax0, orientation='vertical')
    cbar.ax.set_ylabel(r'$T_b$ ($\mu$K)')
    axs[0,0].set_aspect(aspect=1)

    # smoothed
    smoothed_spacestack_gauss = convolve(stackim, params.gauss_kernel)
    vext = np.nanmax(smoothed_spacestack_gauss*1e6)
    c = axs[0,1].pcolormesh(smoothed_spacestack_gauss*1e6, cmap=cmap, vmin=-vext, vmax=vext)
    axs[0,1].plot(xcorners, ycorners, color='0.8', linewidth=4, zorder=10)
    axs[0,1].set_title('Gaussian-smoothed')

    axs[0,1].tick_params(axis='y',
                            labelleft=False,
                            labelright=False,
                            left=False,
                            right=False)
    axs[0,1].tick_params(axis='x',
                         labeltop=False,
                         labelbottom=False,
                         top=False,
                         bottom=False)

    divider = make_axes_locatable(axs[0,1])
    cax0 = divider.new_horizontal(size='5%', pad=0.05)
    fig.add_axes(cax0)
    cbar = fig.colorbar(c, cax=cax0, orientation='vertical')
    cbar.ax.set_ylabel(r'$T_b$ ($\mu$K)')
    axs[0,1].set_aspect(aspect=1)

    if params.freqwidth % 2 == 0:
        freqarr = np.arange(params.freqstackwidth * 2)*31.25e-3 - (params.freqstackwidth-0.5)*31.25e-3
    else:
        freqarr = np.arange(params.freqstackwidth * 2 + 1)*31.25e-3 - (params.freqstackwidth)*31.25e-3
    freqax.step(freqarr, stackspec*1e6,
                color='indigo', zorder=10, where='mid')
    freqax.axhline(0, color='k', ls='--')
    freqax.axvline(0, color='k', ls='--')
    # show which channels contribute to the stack
    freqax.axvline(0 - params.freqwidth / 2 * 31.25e-3, color='0.7', ls=':')
    freqax.axvline(0 + params.freqwidth / 2 * 31.25e-3, color='0.7', ls=':')
    freqax.set_xlabel(r'$\Delta_\nu$ [GHz]')
    freqax.set_ylabel(r'T$_b$ [$\mu$K]')

    if stackresult:
        fig.suptitle('$T_b = {:.3f}\\pm {:.3f}$ $\\mu$K'.format(*stackresult))

    if params.saveplots:
        fig.savefig(params.plotsavepath + '/combinedstackim.png')

    return 0


def catalogue_plotter(catlist, goodcatidx, params):

    fig,axs = plt.subplots(1,3, figsize=(13,4))
    fields = ['Field 1', 'Field 2', 'Field 3']

    for i in range(3):
        fieldz = catlist[i].z[goodcatidx[i]]
        fieldcoord = catlist[i].coords[goodcatidx[i]]

        c = axs[i].scatter(fieldcoord.ra.deg, fieldcoord.dec.deg, c=fieldz, cmap='jet', vmin=2.4, vmax=3.4)
        axs[i].set_xlabel('Dec (deg)')
        axs[i].set_title(fields[i]+' ('+str(len(goodcatidx[i]))+' objects)')

    axs[0].set_ylabel('RA (deg)')
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(c, cax=cax)
    cbar.ax.set_ylabel('Redshift')

    if params.saveplots:
        fig.savefig(params.plotsavepath + '/catalogue_object_distribution.png')

    return 0

def field_catalogue_plotter(cat, goodcatidx, params):

    fig, ax = plt.subplots(1)

    fieldz = cat.z[goodcatidx]
    fieldcoord = cat.coords[goodcatidx]

    c = ax.scatter(fieldcoord.ra.deg, fieldcoord.dec.deg, c=fieldz, cmap='jet', vmin=2.4, vmax=3.4)
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')

    cbar = fig.colorbar(c)
    cbar.ax.set_ylabel('Redshift')

    if params.saveplots:
        fig.savefig(params.plotsavepath + '/catalogue_object_distribution.png')

    return fig

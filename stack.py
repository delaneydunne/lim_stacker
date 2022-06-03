from __future__ import absolute_import, print_function
from .tools import *
import os
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import astropy.constants as const
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel
from astropy.coordinates import SkyCoord

def single_cutout(idx, galcat, comap, params):

    # find gal in each axis, test to make sure it falls into field
    ## freq
    zval = galcat.z[idx]
    nuobs = params.centfreq / (1 + zval)

    diffarr = np.abs(comap.freq - nuobs)
    freqidx = np.argmin(diffarr)

    if diffarr[freqidx] > comap.fstep:
        return None

    fdiff = comap.freq[freqidx] - nuobs


    ## x
    x = galcat.coords[idx].ra.deg
    diffarr = np.abs(comap.ra - x)
    xidx = np.argmin(diffarr)

    if diffarr[xidx] > comap.xstep:
        return None

    xdiff = comap.ra[xidx] - x

    ## y
    y = galcat.coords[idx].dec.deg
    diffarr = np.abs(comap.dec - y)
    yidx = np.argmin(diffarr)

    if diffarr[yidx] > comap.ystep:
        return None

    ydiff = comap.dec[yidx] - y

    # start setting up cutout object if it passes all these tests
    cutout = empty_table()

    # center values of the gal (store for future reference)
    cutout.catidx = idx
    cutout.z = zval
    cutout.coords = galcat.coords[idx]
    cutout.freq = nuobs
    cutout.x = x
    cutout.y = y

    # indices for freq axis
    dfreq = params.freqwidth // 2
    if params.freqwidth % 2 == 0:
        if fdiff < 0:
            freqcutidx = (freqidx - dfreq, freqidx + dfreq)
        else:
            freqcutidx = (freqidx - dfreq + 1, freqidx + dfreq + 1)

    else:
        freqcutidx = (freqidx - dfreq, freqidx + dfreq + 1)
    cutout.freqidx = freqcutidx

    # indices for x axis
    dx = params.xwidth // 2
    if params.xwidth  % 2 == 0:
        if xdiff < 0:
            xcutidx = (xidx - dx, xidx + dx)
        else:
            xcutidx = (xidx - dx + 1, xidx + dx + 1)
    else:
        xcutidx = (xidx - dx, xidx + dx + 1)
    cutout.xidx = xcutidx

    # indices for y axis
    dy = params.ywidth // 2
    if params.ywidth  % 2 == 0:
        if ydiff < 0:
            ycutidx = (yidx - dy, yidx + dy)
        else:
            ycutidx = (yidx - dy + 1, yidx + dy + 1)
    else:
        ycutidx = (yidx - dy, yidx + dy + 1)
    cutout.yidx = ycutidx

    # more checks -- make sure it's not going off the center of the map
    if freqcutidx[0] < 0 or xcutidx[0] < 0 or ycutidx[0] < 0:
        return None
    freqlen, xylen = len(comap.freq), len(comap.x)
    if freqcutidx[1] > freqlen or xcutidx[1] > xylen or ycutidx[1] > xylen:
        return None


    # pull the actual values to stack
    pixval = comap.map[cutout.freqidx[0]:cutout.freqidx[1],
                       cutout.yidx[0]:cutout.yidx[1],
                       cutout.xidx[0]:cutout.xidx[1]]
    rmsval = comap.rms[cutout.freqidx[0]:cutout.freqidx[1],
                       cutout.yidx[0]:cutout.yidx[1],
                       cutout.xidx[0]:cutout.xidx[1]]

    if params.beamscale:
        pixval = pixval*params.beam
        rmsval = rmsval*params.beam

    # if all pixels are masked, lose the whole object
    if np.all(np.isnan(pixval)):
        return None

    # find the actual Tb in the cutout -- weighted average over all axes
    Tbval, Tbrms = weightmean(pixval, rmsval)

    cutout.T = Tbval
    cutout.rms = Tbrms

    # get the bigger cutouts for plotting if desired:
    ## spatial map
    if params.spacestackwidth:
        # same process as above, just wider
        dxy = params.spacestackwidth
        # x-axis
        if params.spacestackwidth  % 2 == 0:
            if xdiff < 0:
                xcutidx = (xidx - dxy, xidx + dxy)
            else:
                xcutidx = (xidx - dxy + 1, xidx + dxy + 1)
        else:
            xcutidx = (xidx - dxy, xidx + dxy + 1)
        cutout.spacexidx = xcutidx

        # y-axis
        if params.spacestackwidth  % 2 == 0:
            if ydiff < 0:
                ycutidx = (yidx - dxy, yidx + dxy)
            else:
                ycutidx = (yidx - dxy + 1, yidx + dxy + 1)
        else:
            ycutidx = (yidx - dxy, yidx + dxy + 1)
        cutout.spaceyidx = ycutidx

        # pad edges of map with nans so you don't have to worry about going off the edge
        padmap = np.pad(comap.map, ((0,0), (dxy,dxy), (dxy,dxy)), 'constant', constant_values=np.nan)
        padrms = np.pad(comap.rms, ((0,0), (dxy,dxy), (dxy,dxy)), 'constant', constant_values=np.nan)

        padxidx = (cutout.spacexidx[0] + dxy, cutout.spacexidx[1] + dxy)
        padyidx = (cutout.spaceyidx[0] + dxy, cutout.spaceyidx[1] + dxy)

        # pull the actual values to stack
        spixval = padmap[cutout.freqidx[0]:cutout.freqidx[1],
                         padyidx[0]:padyidx[1],
                         padxidx[0]:padxidx[1]]
        srmsval = padrms[cutout.freqidx[0]:cutout.freqidx[1],
                         padyidx[0]:padyidx[1],
                         padxidx[0]:padxidx[1]]

        # collapse along freq axis to get a spatial map
        spacestack, rmsspacestack = weightmean(spixval, srmsval, axis=0)
        cutout.spacestack = spacestack
        cutout.spacestackrms = rmsspacestack

    ## spectrum
    if params.freqstackwidth:
        df = params.freqstackwidth
        if params.freqwidth % 2 == 0:
            if fdiff < 0:
                freqcutidx = (freqidx - df, freqidx + df)
            else:
                freqcutidx = (freqidx - df + 1, freqidx + df + 1)

        else:
            freqcutidx = (freqidx - df, freqidx + df + 1)
        cutout.freqfreqidx = freqcutidx

        # pad edges of map with nans so you don't have to worry about going off the edge
        padmap = np.pad(comap.map, ((df,df), (0,0), (0,0)), 'constant', constant_values=np.nan)
        padrms = np.pad(comap.rms, ((df,df), (0,0), (0,0)), 'constant', constant_values=np.nan)

        padfreqidx = (cutout.freqfreqidx[0] + df, cutout.freqfreqidx[1] + df)

        # clip out values to stack
        fpixval = padmap[padfreqidx[0]:padfreqidx[1],
                         cutout.yidx[0]:cutout.yidx[1],
                         cutout.xidx[0]:cutout.xidx[1]]
        frmsval = padrms[padfreqidx[0]:padfreqidx[1],
                         cutout.yidx[0]:cutout.yidx[1],
                         cutout.xidx[0]:cutout.xidx[1]]

        # collapse along spatial axes to get a spectral profile
        freqstack, rmsfreqstack = weightmean(fpixval, frmsval, axis=(1,2))
        cutout.freqstack = freqstack
        cutout.freqstackrms = rmsfreqstack

    return cutout

def field_get_cutouts(comap, galcat, params, field=None):
    """
    wrapper to return all cutouts for a single field
    """

    cutoutlist = []
    for i in range(galcat.nobj):
        cutout = single_cutout(i, galcat, comap, params)

        # if it passed all the tests, keep it
        if cutout:
            if field:
                cutout.field = field

            cutoutlist.append(cutout)

    return cutoutlist

def stacker(maplist, galcatlist, params):
    """
    wrapper to perform a full stack on all available values in the catalogue.
    will plot if desired
    """
    fields = [1,2,3]

    fieldlens = []
    allcutouts = []
    for i in range(len(maplist)):
        fieldcutouts = field_get_cutouts(maplist[i], galcatlist[i], params, field=fields[i])
        fieldlens.append(len(fieldcutouts))
        allcutouts = allcutouts + fieldcutouts

    print(fieldlens)

    # unzip all your cutout objects
    Tvals = []
    rmsvals = []
    catidxs = []
    if params.spacestackwidth:
        spacestack = []
        spacerms = []
    if params.freqstackwidth:
        freqstack = []
        freqrms = []
    for cut in allcutouts:
        Tvals.append(cut.T)
        rmsvals.append(cut.rms)
        catidxs.append(cut.catidx)

        if params.spacestackwidth:
            spacestack.append(cut.spacestack)
            spacerms.append(cut.spacestackrms)

        if params.freqstackwidth:
            freqstack.append(cut.freqstack)
            freqrms.append(cut.freqstackrms)

    # put everything into numpy arrays for ease
    Tvals = np.array(Tvals)
    rmsvals = np.array(rmsvals)
    catidxs = np.array(catidxs)
    if params.spacestackwidth:
        spacestack = np.array(spacestack)
        spacerms = np.array(spacerms)
    if params.freqstackwidth:
        freqstack = np.array(freqstack)
        freqrms = np.array(freqrms)


    # split indices up by field
    fieldcatidx = []
    previdx = 0
    for catlen in fieldlens:
        fieldcatidx.append(catidxs[previdx:catlen+previdx])
        previdx += catlen

    # overall stack for T value
    stacktemp, stackrms = weightmean(Tvals, rmsvals)

    # overall spatial stack
    if params.spacestackwidth:
        stackim, imrms = weightmean(spacestack, spacerms, axis=0)
    else:
        stackim, imrms = None, None

    # overall frequency stack
    if params.freqstackwidth:
        stackspec, specrms = weightmean(freqstack, freqrms, axis=0)
    else:
        stackspec, imrms = None, None

    if params.saveplots:
        # make the directory to store the plots
        os.makedirs(params.savepath, exist_ok=True)

    if params.plotspace:
        spatial_plotter(stackim, params)

    if params.plotfreq:
        spectral_plotter(stackspec, params)

    if params.plotspace and params.plotfreq:
        combined_plotter(stackim, stackspec, params)

    return stacktemp, stackrms, stackim, stackspec, fieldcatidx

""" PLOTTING FUNCTIONS """
def spatial_plotter(stackim, params):

    # corners for the beam rectangle
    if params.xwidth % 2 == 0:
        rectmin = params.spacestackwidth - params.xwidth/2 - 0.5
        rectmax = params.spacestackwidth + params.xwidth/2 - 0.5
    else:
        rectmin = params.spacestackwidth - params.xwidth/2
        rectmax = params.spacestackwidth + params.xwidth/2

    xcorners = (rectmin, rectmin, rectmax, rectmax, rectmin)
    ycorners = (rectmin, rectmax, rectmax, rectmin, rectmin)

    vext = np.max(np.abs([np.min(stackim), np.max(stackim)]))
    vmin,vmax = -vext, vext

    # unsmoothed
    fig, ax = plt.subplots(1)
    c = ax.imshow(stackim*1e6, cmap='PiYG', vmin=vmin, vmax=vmax)
    ax.plot(xcorners, ycorners, color='0.8', linewidth=4, zorder=10)
    cbar = fig.colorbar(c)
    cbar.ax.set_ylabel('Tb (uK)')

    if params.saveplots:
        fig.savefig(params.savepath+'/angularstack_unsmoothed.png')

    # smoothed
    smoothed_spacestack_gauss = convolve(stackim, params.gauss_kernel)

    fig, ax = plt.subplots(1)
    c = ax.imshow(smoothed_spacestack_gauss*1e6, cmap='PiYG', vmin=vmin, vmax=vmax)
    ax.plot(xcorners, ycorners, color='0.8', linewidth=4, zorder=10)
    cbar = fig.colorbar(c)
    cbar.ax.set_ylabel('Tb (uK)')

    if params.saveplots:
        fig.savefig(params.savepath+'/angularstack_smoothed.png')

    return 0

def spectral_plotter(stackspec, params):
    fig, ax = plt.subplots(1)
    if params.freqwidth % 2 == 0:
        freqarr = np.arange(params.freqstackwidth * 2)*31.25e-3 - (params.freqstackwidth-0.5)*31.25e-3
    else:
        freqarr = np.arange(params.freqstackwidth * 2 + 1)*31.25e-3 - (params.freqstackwidth)*31.25e-3
    ax.plot(freqarr, stackspec*1e6,
            color='indigo', zorder=10)
    ax.set_xlabel(r'$\Delta_\nu$ [GHz]')
    ax.set_ylabel(r'T$_b$ [$\mu$K]')
    ax.set_title('Stacked over {} Spatial Pixels'.format((params.xwidth)**2))

    ax.axhline(0, color='k', ls='--')
    ax.axvline(0, color='k', ls='--')

    # show which channels contribute to the stack
    ax.axvline(0 - params.freqwidth / 2, color='0.7', ls=':')
    ax.axvline(0 + params.freqwidth / 2, color='0.7', ls=':')

    if params.saveplots:
        fig.savefig(params.savepath + '/frequencystack.png')

    return 0

def combined_plotter(stackim, stackspec, params):

    # corners for the beam rectangle
    if params.xwidth % 2 == 0:
        rectmin = params.spacestackwidth - params.xwidth/2 - 0.5
        rectmax = params.spacestackwidth + params.xwidth/2 - 0.5
    else:
        rectmin = params.spacestackwidth - params.xwidth/2
        rectmax = params.spacestackwidth + params.xwidth/2

    xcorners = (rectmin, rectmin, rectmax, rectmax, rectmin)
    ycorners = (rectmin, rectmax, rectmax, rectmin, rectmin)

    vext = np.max(np.abs([np.min(stackim), np.max(stackim)]))
    vmin,vmax = -vext, vext

    # plot with all three stack representations
    gs_kw = dict(width_ratios=[1,1], height_ratios=[3,2])
    fig,axs = plt.subplots(2,2, figsize=(7,5), gridspec_kw=gs_kw)
    gs = axs[0,-1].get_gridspec()

    for ax in axs[1,:]:
        ax.remove()

    freqax = fig.add_subplot(gs[-1,:])

    c = axs[0,0].imshow(stackim*1e6, cmap='PiYG', vmin=vmin, vmax=vmax)
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

    # smoothed
    smoothed_spacestack_gauss = convolve(stackim, params.gauss_kernel)
    c = axs[0,1].imshow(smoothed_spacestack_gauss*1e6, cmap='PiYG', vmin=vmin, vmax=vmax)
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

    if params.freqwidth % 2 == 0:
        freqarr = np.arange(params.freqstackwidth * 2)*31.25e-3 - (params.freqstackwidth-0.5)*31.25e-3
    else:
        freqarr = np.arange(params.freqstackwidth * 2 + 1)*31.25e-3 - (params.freqstackwidth)*31.25e-3
    freqax.plot(freqarr, stackspec*1e6,
                color='indigo', zorder=10)
    freqax.axhline(0, color='k', ls='--')
    freqax.axvline(0, color='k', ls='--')
    # show which channels contribute to the stack
    freqax.axvline(0 - params.freqwidth / 2, color='0.7', ls=':')
    freqax.axvline(0 + params.freqwidth / 2, color='0.7', ls=':')
    freqax.set_xlabel(r'$\Delta_\nu$ [GHz]')
    freqax.set_ylabel(r'T$_b$ [$\mu$K]')

    if params.saveplots:
        fig.savefig(params.savepath + '/combinedstackim.png')

    return 0

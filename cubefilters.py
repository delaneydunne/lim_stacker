from __future__ import absolute_import, print_function
from .tools import *
from .stack import *
from .plottools import *
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import astropy.constants as const
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel
from astropy.modeling import models, fitting
import warnings
import csv
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", message="invalid value encountered in power")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")

def remove_cutout_lowmodes(cutout, params, plot=False, plotfit=False):
    """
    function that will return a copy of the passed cutout object with a 2D linear polynomial
    fit to the spatial image and subtracted
    """

    # pull the cutout over the correct number of frequency channels
    try:
        cutim = cutout.spacestack * 1e6
        cutrms = cutout.spacestackrms * 1e6
    except AttributeError:
        aperture_collapse_cubelet_freq(cutout, params)
        cutim = cutout.spacestack * 1e6
        cutrms = cutout.spacestackrms * 1e6

    # mask out the source aperture and the edges -- clip to just the center
    maskrad = int((params.fitmasknbeams - 1) * params.xwidth)
    maskext = np.array([-maskrad, maskrad])
    beamxidx = cutout.xidx - cutout.spacexidx[0] + maskext
    beamyidx = cutout.yidx - cutout.spaceyidx[0] + maskext

    cutim[beamyidx[0]:beamyidx[1], beamxidx[0]:beamxidx[1]] = np.nan
    cutrms[beamyidx[0]:beamyidx[1], beamxidx[0]:beamxidx[1]] = np.nan

    # radius around the center to keep for fitting
    cliprad = int((params.fitnbeams - 1) * params.xwidth)
    clipext = np.array([-cliprad, cliprad])
    clipxidx, clipyidx = beamxidx + clipext, beamyidx + clipext

    # final cutout to fit
    # has to go into uk so as to not cause problems
    cutim = cutim[clipyidx[0]:clipyidx[1], clipxidx[0]:clipxidx[1]]
    cutrms = cutrms[clipyidx[0]:clipyidx[1], clipxidx[0]:clipxidx[1]]

    # Fit the data using astropy.modeling -- set up a 2d polynomial to fit
    p_init = models.Polynomial2D(degree=1)
    fit_p = fitting.LevMarLSQFitter()
    # x and y axes
    y,x = np.mgrid[:len(cutim), :len(cutim)]

    # mask nans because the fitter can't deal with them
    mask = np.isfinite(cutim)
    p = fit_p(p_init, x[mask], y[mask], cutim[mask], weights=1/cutrms[mask])

    # if the mean in these central channels is way off then assume the whole
    # cutout is bad
    # also cutting on slopes > 10 in either of the two gradient directions
    if np.abs(p.c0_0) > params.fitmeanlimit or np.abs(p.c1_0) > 10 or np.abs(p.c0_1) > 10:
        return None

    if plotfit:
        fig,axs = plt.subplots(1,4, sharey=True, sharex=True)
        vl,vu = simlims(cutim)
        axs[0].pcolormesh(cutim, vmin=vl, vmax=vu, cmap='PiYG_r')
        axs[0].set_title('Raw Cutout')
        vl,vu = simlims(p(x,y))
        axs[1].pcolormesh(p(x, y), vmin=vl, vmax=vu, cmap='PiYG_r')
        axs[1].set_title('Linear 2D Fit')
        vl,vu = simlims(cutim - p(x,y))
        axs[2].pcolormesh(cutim - p(x, y), vmin=vl, vmax=vu, cmap='PiYG_r')
        axs[2].set_title('Residual')
        axs[3].pcolormesh(1/cutrms, cmap='PiYG_r')
        axs[3].set_title('Weighting')

        for ax in axs:
            ax.set_aspect(aspect=1)

    # subtract this polynomial from the full cutout and add it into the cutout object
    fully, fullx = np.mgrid[:len(cutout.spacestack), :len(cutout.spacestack)]
    fullcutim = cutout.spacestack

    newcutout = cutout.copy()
    newcutout.polyfit = p(fullx, fully) / 1e6
    newcutout.polyfitmodel = p.parameters
    newcutout.spacestack = fullcutim - newcutout.polyfit

    # also subtract from the full cubelet
    fullcube = cutout.cubestack
    cubepolyfit = np.tile(newcutout.polyfit, (fullcube.shape[0],1,1))
    newcutout.cubestack = fullcube - cubepolyfit

    if plot:
        fig,axs = plt.subplots(1,3, sharey=True, sharex=True)
        vl,vu = simlims(fullcutim)
        axs[0].pcolormesh(fullcutim, vmin=vl, vmax=vu, cmap='PiYG_r')
        axs[0].set_title('Raw Cutout')
        print(vl,vu)
        vl,vu = simlims(newcutout.polyfit)
        axs[1].pcolormesh(newcutout.polyfit, vmin=vl, vmax=vu, cmap='PiYG_r')
        axs[1].set_title('Linear 2D Fit')
        print(vl,vu)
        vl,vu = simlims(newcutout.spacestack)
        axs[2].pcolormesh(newcutout.spacestack, vmin=vl, vmax=vu, cmap='PiYG_r')
        axs[2].set_title('Residual')
        print(vl,vu)

        for ax in axs:
            ax.set_aspect(aspect=1)

    return newcutout


def remove_cutout_chanmean(cutout, params, plot=False, plotfit=False):
    """
    function to, for a given cutout, find the region around the source spatially
    but not including the actual source, find the mean value of each channel, and
    subtract those means from the cutout
    """

    # cubelet
    cutim = copy.deepcopy(cutout.cubestack)
    cutrms = copy.deepcopy(cutout.cubestackrms)

    # mask out the source aperture and the edges -- clip to just the center
    beamxidx = cutout.xidx - cutout.spacexidx[0]
    beamyidx = cutout.yidx - cutout.spaceyidx[0]
    beamfidx = cutout.freqidx - cutout.freqfreqidx[0]

    cutim[beamfidx[0]:beamfidx[1],
          beamyidx[0]:beamyidx[1],
          beamxidx[0]:beamxidx[1]] = np.nan
    cutrms[beamfidx[0]:beamfidx[1],
           beamyidx[0]:beamyidx[1],
           beamxidx[0]:beamxidx[1]] = np.nan

    # radius around the center to keep for fitting in space (keep all freq channels)
    cliprad = (params.fitnbeams - 1) * params.xwidth
    clipext = np.array([-cliprad, cliprad])
    clipxidx, clipyidx = beamxidx + clipext, beamyidx + clipext

    # final cutout to fit
    cutim = cutim[:, clipyidx[0]:clipyidx[1], clipxidx[0]:clipxidx[1]]
    cutrms = cutrms[:, clipyidx[0]:clipyidx[1], clipxidx[0]:clipxidx[1]]

    # use the variance-weighted mean to find a mean value for each channel in the cube
    chanmeans, _ = weightmean(cutim, cutrms, axis=(1,2))

    # check the mean channels to make sure they aren't too crazy
    if np.all(chanmeans[beamfidx[0]:beamfidx[1]] > params.fitmeanlimit/1e6):
        return None

    # subtract off the means and store in a new cutout object
    newcutout = cutout.copy()
    newcutout.chanmeans = chanmeans
    # ugly flipping bc the freq axis is on the wrong side
    newcutout.cubestack = (cutout.cubestack.T - chanmeans).T

    return newcutout


def remove_cutout_spectral_mean(cutout, params, plot=False):
    """
    function to, for a given cutout, find the region around the source spectrally
    (not including the actual source), find the global mean value of this nearby
    spectrum, and subtract that mean from the cutout
    """

    # find the channels to mask
    apidx = cutout.freqidx - cutout.freqfreqidx[0]
    if params.freqmaskwidth > 1:
        maskrad = (params.freqmaskwidth - 1) * params.freqwidth
        maskext = np.array([-maskrad, maskrad])
        maskfidx = apidx + maskext
    else:
        maskfidx = apidx

    try:
        # copy of the spectrum (there was weird overwriting stuff if not)
        maskarr = copy.deepcopy(cutout.freqstack)
        maskrms = copy.deepcopy(cutout.freqstackrms)
    except AttributeError:
        aperture_collapse_cubelet_space(cutout, params)
        # copy of the spectrum (there was weird overwriting stuff if not)
        maskarr = copy.deepcopy(cutout.freqstack)
        maskrms = copy.deepcopy(cutout.freqstackrms)

    # mask the channels that probably contain the source
    maskarr[maskfidx[0]:maskfidx[1]] = np.nan
    maskrms[maskfidx[0]:maskfidx[1]] = np.nan

    # outer channels to exclude (this is a passed parameter for now but maybe base it on the rms?)
    cliprad = (params.frequsewidth - 1) * params.freqwidth
    clipext = np.array([-cliprad, cliprad])
    clipidx = apidx + clipext

    # clip off the faraway channels
    maskarr = maskarr[clipidx[0]:clipidx[1]]
    maskrms = maskrms[clipidx[0]:clipidx[1]]

    # noise-weighted mean value
    freqmean, _ = weightmean(maskarr, maskrms)

    # new cutout object with the mean subtracted
    newcutout = cutout.copy()
    newcutout.freqmean = freqmean
    newcutout.cubestack = cutout.cubestack - freqmean

    # diagnostic plotter
    if plot:
        # frequency axis to plot against with actual GHz values
        if params.freqwidth % 2 == 0:
            freqarr = np.arange(params.freqstackwidth * 2)*31.25e-3 - (params.freqstackwidth-0.5)*31.25e-3
        else:
            freqarr = np.arange(params.freqstackwidth * 2 + 1)*31.25e-3 - (params.freqstackwidth)*31.25e-3

        maskfreqarr = copy.deepcopy(freqarr)[clipidx[0]:clipidx[1]]

        fig, ax = plt.subplots(1, tight_layout=True)
        plt.style.use('seaborn-talk')

        # original array
        ax.step(freqarr, testcut.freqstack*1e6, color='0.5', where='mid')
        # part of the array from which the mean value is calculated
        ax.step(maskfreqarr, maskarr*1e6, color='indigo', where='mid')

        # mean value
        ax.axhline(freqmean, color='k', ls='--')

        # lines to show the demarcations
        ax.axvline(freqarr[maskfidx[0]], color='0.8')
        ax.axvline(freqarr[maskfidx[1]], color='0.8')
        ax.axvline(freqarr[clipidx[0]], color='0.8')
        ax.axvline(freqarr[clipidx[1]], color='0.8')


        ax.set_xlabel(r'$\Delta_\nu$ [GHz]')
        ax.set_ylabel(r'T$_b$ [$\mu$K]')

    return newcutout

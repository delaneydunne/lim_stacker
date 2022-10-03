from __future__ import absolute_import, print_function
from .tools import *
from .plottools import *
import os
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


# standard COMAP cosmology
cosmo = FlatLambdaCDM(H0=70*u.km / (u.Mpc*u.s), Om0=0.286, Ob0=0.047)

""" CUTOUT-SPECIFIC FUNCTIONS """
def single_cutout(idx, galcat, comap, params):

    # find gal in each axis, test to make sure it falls into field
    ## freq
    zval = galcat.z[idx]
    nuobs = params.centfreq / (1 + zval)
    if nuobs < np.min(comap.freq) or nuobs > np.max(comap.freq + comap.fstep):
        return None
    freqidx = np.max(np.where(comap.freq < nuobs))
    if np.abs(nuobs - comap.freq[freqidx]) < comap.fstep / 2:
        fdiff = -1
    else:
        fdiff = 1

    x = galcat.coords[idx].ra.deg
    if x < np.min(comap.ra) or x > np.max(comap.ra + comap.xstep):
        return None
    xidx = np.max(np.where(comap.ra < x))
    if np.abs(x - comap.ra[xidx]) < comap.xstep / 2:
        xdiff = -1
    else:
        xdiff = 1

    y = galcat.coords[idx].dec.deg
    if y < np.min(comap.dec) or y > np.max(comap.dec + comap.ystep):
        return None
    yidx = np.max(np.where(comap.dec < y))
    if np.abs(y - comap.dec[yidx]) < comap.ystep / 2:
        ydiff = -1
    else:
        ydiff = 1

    # start setting up cutout object if it passes all these tests
    cutout = empty_table()

    # center values of the gal (store for future reference)
    cutout.catidx = idx
    cutout.z = zval
    cutout.coords = galcat.coords[idx]
    cutout.freq = nuobs
    cutout.x = x
    cutout.y = y

    # index the actual aperture to be stacked from the cutout
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

    # get the bigger cutouts for plotting if desired:
    ## cubelet
    if params.spacestackwidth and params.freqstackwidth:

        # same process as above, just wider
        df = params.freqstackwidth
        if params.freqwidth % 2 == 0:
            if fdiff < 0:
                freqcutidx = (freqidx - df, freqidx + df)
            else:
                freqcutidx = (freqidx - df + 1, freqidx + df + 1)

        else:
            freqcutidx = (freqidx - df, freqidx + df + 1)
        cutout.freqfreqidx = freqcutidx

        dxy = params.spacestackwidth
        # x-axis
        if params.xwidth  % 2 == 0:
            if xdiff < 0:
                xcutidx = (xidx - dxy, xidx + dxy)
            else:
                xcutidx = (xidx - dxy + 1, xidx + dxy + 1)
        else:
            xcutidx = (xidx - dxy, xidx + dxy + 1)
        cutout.spacexidx = xcutidx

        # y-axis
        if params.ywidth  % 2 == 0:
            if ydiff < 0:
                ycutidx = (yidx - dxy, yidx + dxy)
            else:
                ycutidx = (yidx - dxy + 1, yidx + dxy + 1)
        else:
            ycutidx = (yidx - dxy, yidx + dxy + 1)
        cutout.spaceyidx = ycutidx

        # pad edges of map with nans so you don't have to worry about going off the edge
        padmap = np.pad(comap.map, ((df,df), (dxy,dxy), (dxy,dxy)), 'constant', constant_values=np.nan)
        padrms = np.pad(comap.rms, ((df,df), (dxy,dxy), (dxy,dxy)), 'constant', constant_values=np.nan)

        padfreqidx = (cutout.freqfreqidx[0] + df, cutout.freqfreqidx[1] + df)
        padxidx = (cutout.spacexidx[0] + dxy, cutout.spacexidx[1] + dxy)
        padyidx = (cutout.spaceyidx[0] + dxy, cutout.spaceyidx[1] + dxy)

        # pull the actual values to stack
        cpixval = padmap[padfreqidx[0]:padfreqidx[1],
                         padyidx[0]:padyidx[1],
                         padxidx[0]:padxidx[1]]
        crmsval = padrms[padfreqidx[0]:padfreqidx[1],
                         padyidx[0]:padyidx[1],
                         padxidx[0]:padxidx[1]]

        # rotate randomly
        if params.rotate:
            cutout.rotangle = params.rng.integers(4) + 1
            cpixval = np.rot90(cpixval, cutout.rotangle, axes=(1,2))
            crmsval = np.rot90(crmsval, cutout.rotangle, axes=(1,2))

        cutout.cubestack = cpixval
        cutout.cubestackrms = crmsval

    ## spatial map
    elif params.spacestackwidth and not params.freqstackwidth:
        # just do the 2d spatial image
        # same process as above, just wider
        dxy = params.spacestackwidth
        # x-axis
        if params.xwidth  % 2 == 0:
            if xdiff < 0:
                xcutidx = (xidx - dxy, xidx + dxy)
            else:
                xcutidx = (xidx - dxy + 1, xidx + dxy + 1)
        else:
            xcutidx = (xidx - dxy, xidx + dxy + 1)
        cutout.spacexidx = xcutidx

        # y-axis
        if params.ywidth  % 2 == 0:
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
        # spacestack = np.nansum(spixval, axis=0)
        # rmsspacestack = np.sqrt(np.nansum(srmsval**2, axis=0))

        # rotate randomly
        if params.rotate:
            cutout.rotangle = params.rng.integers(4) + 1
            spacestack = np.rot90(spacestack, cutout.rotangle)
            rmsspacestack = np.rot90(rmsspacestack, cutout.rotangle)

        cutout.spacestack = spacestack
        cutout.spacestackrms = rmsspacestack

    ## spectrum
    elif params.freqstackwidth and not params.spacestackwidth:
        # just the 1d spectrum along the frequency axis
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
        # freqstack = np.nansum(fpixval, axis=(1,2))
        # rmsfreqstack = np.sqrt(np.nansum(frmsval**2, axis=(1,2)))
        cutout.freqstack = freqstack
        cutout.freqstackrms = rmsfreqstack

    # pull the actual values to stack
    pixval = comap.map[cutout.freqidx[0]:cutout.freqidx[1],
                       cutout.yidx[0]:cutout.yidx[1],
                       cutout.xidx[0]:cutout.xidx[1]]
    rmsval = comap.rms[cutout.freqidx[0]:cutout.freqidx[1],
                       cutout.yidx[0]:cutout.yidx[1],
                       cutout.xidx[0]:cutout.xidx[1]]

    # if all pixels are masked, lose the whole object
    if np.all(np.isnan(pixval)):
        return None

    # find the actual Tb in the cutout -- weighted average over all axes
    Tbval, Tbrms = weightmean(pixval, rmsval)
    if np.isnan(Tbval):
        return None

    # subtract the low-order modes
    if params.lowmodefilter:
        cutout = remove_cutout_lowmodes(cutout, params)

    # Tbval = np.nansum(pixval)
    # Tbrms = np.sqrt(np.nansum(rmsval**2))

    cutout.T = Tbval
    cutout.rms = Tbrms

    return cutout

# convenience functions
def aperture_collapse_cubelet_freq(cutout, params):
    """
    take a 3D cubelet cutout and collapse it along the frequency axis to be an average over the
    stack aperture frequency channels
    """

    # indexes of the channels to include
    lcfidx = (cutout.cubestack.shape[0] - params.freqwidth) // 2
    cfidx = (lcfidx, lcfidx + params.freqwidth)

    # collapsed image
    cutim, imrms = weightmean(cutout.cubestack[cfidx[0]:cfidx[1],:,:],
                                 cutout.cubestackrms[cfidx[0]:cfidx[1],:,:], axis=0)

    cutout.spacestack = cutim
    cutout.spacestackrms = imrms

    return

def aperture_collapse_cubelet_space(cutout, params):
    """
    take a 3D cubelet cutout and collapse it along the spatial axis to be an average over the stack
    aperture spaxels (ie make a spectrum)
    """

    # indices of the x and y axes
    beamxidx = cutout.xidx - cutout.spacexidx[0]
    beamyidx = cutout.yidx - cutout.spaceyidx[0]

    # clip out values to stack
    fpixval = cutout.cubestack[:, beamyidx[0]:beamyidx[1], beamxidx[0]:beamxidx[1]]
    frmsval = cutout.cubestackrms[:, beamyidx[0]:beamyidx[1], beamxidx[0]:beamxidx[1]]

    # collapse along spatial axes to get a spectral profile
    freqstack, rmsfreqstack = weightmean(fpixval, frmsval, axis=(1,2))

    cutout.freqstack = freqstack
    cutout.freqstackrms = rmsfreqstack

    return

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
    cutim = cutout.cubestack
    cutrms = cutout.cubestackrms

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
    chanmeans, _ = st.weightmean(cutim, cutrms, axis=(1,2))

    # subtract off the means and store in a new cutout object
    newcutout = cutout.copy()
    newcutout.chanmeans = chanmeans
    # ugly flipping bc the freq axis is on the wrong side
    newcutout.cubestack = (cutout.cubestack.T - chanmeans).T

    return newcutout


def field_get_cutouts(comap, galcat, params, field=None, goalnobj=None):
    """
    wrapper to return all cutouts for a single field
    """
    ti = 0
    # if we're keeping track of the number of cutouts
    if goalnobj:
        field_nobj = 0
    cubestack, cuberms = None, None
    cutoutlist = []
    for i in range(galcat.nobj):
        cutout = single_cutout(i, galcat, comap, params)

        # if it passed all the tests, keep it
        if cutout:
            if field:
                cutout.field = field

            if params.cubelet:
                if ti == 0:
                    cubestack = cutout.cubestack
                    cuberms = cutout.cubestackrms
                    # delete the 3d arrays
                    cutout.__delattr__('cubestack')
                    cutout.__delattr__('cubestackrms')
                    ti = 1
                else:
                    scstack = np.stack((cubestack, cutout.cubestack))
                    scrms = np.stack((cuberms, cutout.cubestackrms))
                    cubestack, cuberms = weightmean(scstack, scrms, axis=0)
                    # delete the 3d arrays
                    cutout.__delattr__('cubestack')
                    cutout.__delattr__('cubestackrms')

            cutoutlist.append(cutout)
            if goalnobj:
                field_nobj += 1

                if field_nobj == goalnobj:
                    if params.verbose:
                        print("Hit goal number of {} cutouts".format(goalnobj))
                    if params.cubelet:
                        return cutoutlist, cubestack, cuberms
                    else:
                        return cutoutlist

        if params.verbose:
            if i % 100 == 0:
                print('   done {} of {} cutouts in this field'.format(i, galcat.nobj))

    if params.cubelet:
        return cutoutlist, cubestack, cuberms
    else:
        return cutoutlist

def stacker(maplist, galcatlist, params, cmap='PiYG_r'):
    """
    wrapper to perform a full stack on all available values in the catalogue.
    will plot if desired
    """
    fields = [1,2,3]

    # for simulations -- if the stacker should stop after a certain number
    # of cutouts. set this up to be robust against per-field or total vals
    if params.goalnumcutouts:
        if isinstance(params.goalnumcutouts, (int, float)):
            numcutoutlist = [params.goalnumcutouts // len(maplist),
                             params.goalnumcutouts // len(maplist),
                             params.goalnumcutouts // len(maplist)]
        else:
            numcutoutlist = params.goalnumcutouts
    else:
        numcutoutlist = [None, None, None]

    # dict to store stacked values
    outputvals = {}

    fieldlens = []
    allcutouts = []
    if params.cubelet:
        cubestacks = []
        cubermss = []
    for i in range(len(maplist)):
        if params.cubelet:
            fieldcutouts, fieldcubestack, fieldcuberms = field_get_cutouts(maplist[i],
                                                                           galcatlist[i],
                                                                           params,
                                                                           field=fields[i],
                                                                           goalnobj=numcutoutlist[i])
            if isinstance(fieldcubestack, np.ndarray):
                cubestacks.append(fieldcubestack)
                cubermss.append(fieldcuberms)
        else:
            fieldcutouts = field_get_cutouts(maplist[i], galcatlist[i], params,
                                             field=fields[i],
                                             goalnobj=numcutoutlist[i])
        fieldlens.append(len(fieldcutouts))
        allcutouts = allcutouts + fieldcutouts

        if params.verbose:
            print('Field {} complete'.format(fields[i]))

    # mean together the individual field stacks if that had to be done separately
    if params.cubelet:
        cubestacks, cubermss = np.stack(cubestacks), np.stack(cubermss)
        cubestack, cuberms = weightmean(cubestacks, cubermss, axis=0)

    nobj = np.sum(fieldlens)
    outputvals['nobj'] = nobj
    if params.verbose:
        print('number of objects in each field is:')
        print('   field 1:{}'.format(fieldlens[0]))
        print('   field 2:{}'.format(fieldlens[1]))
        print('   field 3:{}'.format(fieldlens[2]))
        print('for a total number of {} objects'.format(nobj))

    # unzip all your cutout objects
    cutlistdict = unzip(allcutouts)

    # put into physical units if requested
    if params.obsunits:
        allou = observer_units(cutlistdict['T'], cutlistdict['rms'], cutlistdict['z'],
                               cutlistdict['freq'], params)


        linelumstack, dlinelumstack = weightmean(allou['L'], allou['dL'])
        rhoh2stack, drhoh2stack = weightmean(allou['rho'], allou['drho'])

        outputvals['linelum'], outputvals['dlinelum'] = linelumstack, dlinelumstack
        outputvals['rhoh2'], outputvals['drhoh2'] = rhoh2stack, drhoh2stack
        outputvals['nuobs_mean'], outputvals['z_mean'] = allou['nuobs_mean'], allou['z_mean']

    # split indices up by field for easy access later
    fieldcatidx = []
    previdx = 0
    for catlen in fieldlens:
        fieldcatidx.append(cutlistdict['catidx'][previdx:catlen+previdx])
        previdx += catlen

    # overall stack for T value
    stacktemp, stackrms = weightmean(cutlistdict['T'], cutlistdict['rms'])
    outputvals['T'], outputvals['rms'] = stacktemp, stackrms

    """ EXTRA STACKS """
    # if cubelets returned, do all three stack versions
    if params.spacestackwidth and params.freqstackwidth:
        if params.cubelet:
            # only want the beam for the axes that aren't being shown
            lcfidx = (cubestack.shape[0] - params.freqwidth) // 2
            cfidx = (lcfidx, lcfidx + params.freqwidth)

            lcxidx = (cubestack.shape[1] - params.xwidth) // 2
            cxidx = (lcxidx, lcxidx + params.xwidth)

            stackim, imrms = weightmean(cubestack[cfidx[0]:cfidx[1],:,:],
                                        cuberms[cfidx[0]:cfidx[1],:,:], axis=0)
            stackspec, specrms = weightmean(cubestack[:,cxidx[0]:cxidx[1],cxidx[0]:cxidx[1]],
                                            cuberms[:,cxidx[0]:cxidx[1],cxidx[0]:cxidx[1]],
                                            axis=(1,2))
        else:
            stackim, imrms = weightmean(cutlistdict['cubestack'][:,cfidx[0]:cfidx[1],:,:],
                                        cutlistdict['cubestackrms'][:,cfidx[0]:cfidx[1],:,:],
                                        axis=(0,1))
            stackspec, specrms = weightmean(cutlistdict['cubestack'][:,:,cxidx[0]:cxidx[1],cxidx[0]:cxidx[1]],
                                            cutlistdict['cubestackrms'][:,:,cxidx[0]:cxidx[1],cxidx[0]:cxidx[1]],
                                            axis=(0,2,3))
    elif params.spacestackwidth and not params.freqstackwidth:
        # overall spatial stack
        stackim, imrms = weightmean(cutlistdict['spacestack'], cutlistdict['spacestackrms'], axis=0)
        stackspec, specrms = None, None
    elif params.freqstackwidth and not params.spacestackwidth:
        # overall frequency stack
        stackspec, specrms = weightmean(cutlistdict['freqstack'], cutlistdict['freqstackrms'], axis=0)
        stackim, imrms = None, None
    else:
        stackim, imrms = None, None
        stackspec, specrms = None, None


    """ PLOTS """
    if params.saveplots:
        # just in case this was done elsewhere in the file structure
        params.make_output_pathnames()
        catalogue_overplotter(galcatlist, maplist, fieldcatidx, params)

    if params.spacestackwidth and params.plotspace:
        spatial_plotter(stackim, params, cmap=cmap)

    if params.freqstackwidth and params.plotfreq:
        spectral_plotter(stackspec, params)

    if params.spacestackwidth and params.freqstackwidth and params.plotspace and params.plotfreq:
        combined_plotter(stackim, stackspec, params, cmap=cmap, stackresult=(stacktemp*1e6,stackrms*1e6))

    if params.plotcubelet:
        cubelet_plotter(cubestack, cuberms, params)

    """ SAVE DATA """
    if params.savedata:
        # save the output values
        ovalfile = params.datasavepath + '/output_values.csv'
        # strip the values of their units before saving them (otherwise really annoying
        # to read out on the other end)
        outputvals_nu = dict_saver(outputvals, ovalfile)

        idxfile = params.datasavepath + '/included_cat_indices.npz'
        np.savez(idxfile, field1=fieldcatidx[0], field2=fieldcatidx[1], field3=fieldcatidx[2])

        if params.spacestackwidth:
            imfile = params.datasavepath + '/stacked_image.npz'
            np.savez(imfile, T=stackim, rms=imrms)

        if params.freqstackwidth:
            specfile = params.datasavepath + '/stacked_spectrum.npz'
            np.savez(specfile, T=stackspec, rms=specrms)

        if params.cubelet:
            cubefile = params.datasavepath + '/stacked_3d_cubelet.npz'
            np.savez(cubefile, T=cubestack, rms=cuberms)

    if params.cubelet:
        return outputvals, stackim, stackspec, fieldcatidx, cubestack, cuberms
    else:
        return outputvals, stackim, stackspec, fieldcatidx

def field_stacker(comap, galcat, params, cmap='PiYG_r', field=None):
    """
    wrapper to perform a full stack on all available values in the catalogue.
    will plot if desired
    """

    # set up for rotating each cutout randomly if that's set to happen
    if params.rotate:
        params.rng = np.random.default_rng(params.rotseed)

    # dict to store stacked values
    outputvals = {}

    # if the stacker should stop after a certain number of cutouts are made
    if params.goalnumcutouts:
        if isinstance(params.goalnumcutouts, (list, tuple, np.ndarray)):
            warnings.warn('List of goalncutouts given but only stacking one field', RuntimeWarning)
            params.goalnumcutouts = params.goalnumcutouts[0]

    # get the cutouts for the field
    if params.cubelet:
        allcutouts, cubestack, cuberms = field_get_cutouts(comap, galcat, params,
                                                           field=field,
                                                           goalnobj = params.goalnumcutouts)
    else:
        allcutouts = field_get_cutouts(comap, galcat, params, field=field,
                                       goalnobj = params.goalnumcutouts)

    if params.verbose:
        print('Field complete')

    # number of objects
    nobj = fieldlen = len(allcutouts)
    outputvals['nobj'] = nobj
    if params.verbose:
        print('number of objects in field is: {}'.format(fieldlen))

    # unzip all your cutout objects
    cutlistdict = unzip(allcutouts)

    # put into physical units if requested
    if params.obsunits:
        allou = observer_units(cutlistdict['T'], cutlistdict['rms'], cutlistdict['z'],
                               cutlistdict['freq'], params)


        linelumstack, dlinelumstack = weightmean(allou['L'], allou['dL'])
        rhoh2stack, drhoh2stack = weightmean(allou['rho'], allou['drho'])

        outputvals['linelum'], outputvals['dlinelum'] = linelumstack, dlinelumstack
        outputvals['rhoh2'], outputvals['drhoh2'] = rhoh2stack, drhoh2stack
        outputvals['nuobs_mean'], outputvals['z_mean'] = allou['nuobs_mean'], allou['z_mean']

    # split indices up by field for easy access later
    fieldcatidx = cutlistdict['catidx']

    # overall stack for T value
    stacktemp, stackrms = weightmean(cutlistdict['T'], cutlistdict['rms'])
    outputvals['T'], outputvals['rms'] = stacktemp, stackrms

    """ EXTRA STACKS """
    # if cubelets returned, do all three stack versions
    if params.spacestackwidth and params.freqstackwidth:
        if params.cubelet:
            # only want the beam for the axes that aren't being shown
            lcfidx = (cubestack.shape[0] - params.freqwidth) // 2
            cfidx = (lcfidx, lcfidx + params.freqwidth)

            lcxidx = (cubestack.shape[1] - params.xwidth) // 2
            cxidx = (lcxidx, lcxidx + params.xwidth)

            stackim, imrms = weightmean(cubestack[cfidx[0]:cfidx[1],:,:],
                                        cuberms[cfidx[0]:cfidx[1],:,:], axis=0)
            stackspec, specrms = weightmean(cubestack[:,cxidx[0]:cxidx[1],cxidx[0]:cxidx[1]],
                                            cuberms[:,cxidx[0]:cxidx[1],cxidx[0]:cxidx[1]],
                                            axis=(1,2))
        else:
            stackim, imrms = weightmean(cutlistdict['cubestack'][:,cfidx[0]:cfidx[1],:,:],
                                        cutlistdict['cubestackrms'][:,cfidx[0]:cfidx[1],:,:],
                                        axis=(0,1))
            stackspec, specrms = weightmean(cutlistdict['cubestack'][:,:,cxidx[0]:cxidx[1],cxidx[0]:cxidx[1]],
                                            cutlistdict['cubestackrms'][:,:,cxidx[0]:cxidx[1],cxidx[0]:cxidx[1]],
                                            axis=(0,2,3))
    elif params.spacestackwidth and not params.freqstackwidth:
        # overall spatial stack
        stackim, imrms = weightmean(cutlistdict['spacestack'], cutlistdict['spacestackrms'], axis=0)
        stackspec, specrms = None, None
    elif params.freqstackwidth and not params.spacestackwidth:
        # overall frequency stack
        stackspec, specrms = weightmean(cutlistdict['freqstack'], cutlistdict['freqstackrms'], axis=0)
        stackim, imrms = None, None
    else:
        stackim, imrms = None, None
        stackspec, specrms = None, None


    """ PLOTS """
    if params.saveplots:
        # just in case this was accidentally done elsewhere in the dir structure
        params.make_output_pathnames()
        field_catalogue_plotter(galcat, fieldcatidx, params)

    if params.spacestackwidth and params.plotspace:
        spatial_plotter(stackim, params, cmap=cmap)

    if params.freqstackwidth and params.plotfreq:
        spectral_plotter(stackspec, params)

    if params.spacestackwidth and params.freqstackwidth and params.plotspace and params.plotfreq:
        combined_plotter(stackim, stackspec, params, cmap=cmap, stackresult=(stacktemp*1e6,stackrms*1e6))

    if params.plotcubelet:
        cubelet_plotter(cubestack, cuberms, params)

    """ SAVE DATA """
    if params.savedata:
        # save the output values
        ovalfile = params.datasavepath + '/output_values.csv'
        # strip the values of their units before saving them (otherwise really annoying
        # to read out on the other end)
        outputvals_nu = dict_saver(outputvals, ovalfile)

        idxfile = params.datasavepath + '/included_cat_indices.npz'
        np.savez(idxfile, field1=fieldcatidx[0], field2=fieldcatidx[1], field3=fieldcatidx[2])

        if params.spacestackwidth:
            imfile = params.datasavepath + '/stacked_image.npz'
            np.savez(imfile, T=stackim, rms=imrms)

        if params.freqstackwidth:
            specfile = params.datasavepath + '/stacked_spectrum.npz'
            np.savez(specfile, T=stackspec, rms=specrms)

        if params.cubelet:
            cubefile = params.datasavepath + '/stacked_3d_cubelet.npz'
            np.savez(cubefile, T=cubestack, rms=cuberms)

    if params.cubelet:
        return outputvals, stackim, stackspec, fieldcatidx, cubestack, cuberms
    else:
        return outputvals, stackim, stackspec, fieldcatidx

def observer_units(Tvals, rmsvals, zvals, nuobsvals, params):
    """
    unit change to physical units
    """

    # main beam to full beam correction
    Tvals = Tvals / 0.7
    rmsvals = rmsvals / 0.7

    # actual beam FWHP is a function of frequency - listed values are 4.9,4.5,4.4 arcmin at 26, 30, 34GHz
    # set up a function to interpolate on
    # beamthetavals = np.array([4.9,4.5,4.4])
    # beamthetafreqs = np.array([26, 30, 34])

    # beamthetas = np.interp(nuobsvals, beamthetafreqs, beamthetavals)*u.arcmin
    # omega_Bs = 1.33*beamthetas**2

    # the 'beam' here is actually the stack aperture size
    beamsigma = params.xwidth / 2 * 2*u.arcmin
    omega_B = (2 / np.sqrt(2*np.log(2)))*np.pi*beamsigma**2

    nuobsvals = nuobsvals*u.GHz
    meannuobs = np.nanmean(nuobsvals)

    onesiglimvals = Tvals + rmsvals

    Sact = (Tvals*u.K).to(u.Jy, equivalencies=u.brightness_temperature(nuobsvals, omega_B))
    Ssig = (onesiglimvals*u.K).to(u.Jy, equivalencies=u.brightness_temperature(nuobsvals, omega_B))
    Srms = (rmsvals*u.K).to(u.Jy, equivalencies=u.brightness_temperature(nuobsvals, omega_B))

    # channel widths in km/s
    delnus = (31.25*u.MHz*params.freqwidth / nuobsvals * const.c).to(u.km/u.s)

    # luminosity distances in Mpc
    DLs = cosmo.luminosity_distance(zvals)

    # line luminosities
    linelumact = const.c**2/(2*const.k_B)*Sact*delnus*DLs**2/ (nuobsvals**2*(1+zvals)**3)
    linelumsig = const.c**2/(2*const.k_B)*Ssig*delnus*DLs**2/ (nuobsvals**2*(1+zvals)**3)
    linelumrms = const.c**2/(2*const.k_B)*Srms*delnus*DLs**2/ (nuobsvals**2*(1+zvals)**3)


    # fixing units
    beamvalobs = linelumact.to(u.K*u.km/u.s*u.pc**2)
    beamvalslim = linelumsig.to(u.K*u.km/u.s*u.pc**2)
    beamrmsobs = linelumrms.to(u.K*u.km/u.s*u.pc**2)

    # rho H2:
    alphaco = 3.6*u.Msun / (u.K * u.km/u.s*u.pc**2)
    mh2us = (linelumact + 2*linelumrms) * alphaco
    mh2obs = linelumact * alphaco
    mh2rms = linelumrms * alphaco

    nu1 = nuobsvals - params.freqwidth/2*0.0625*u.GHz
    nu2 = nuobsvals + params.freqwidth/2*0.0625*u.GHz

    z = (params.centfreq*u.GHz - nuobsvals) / nuobsvals
    z1 = (params.centfreq*u.GHz - nu1) / nu1
    z2 = (params.centfreq*u.GHz - nu2) / nu2
    meanz = np.nanmean(z)

    distdiff = cosmo.luminosity_distance(z1) - cosmo.luminosity_distance(z2)

    # proper volume at each voxel
    volus = ((cosmo.kpc_proper_per_arcmin(z1) * params.xwidth*2*u.arcmin).to(u.Mpc))**2 * distdiff

    rhous = mh2us / volus
    rhousobs = mh2obs / volus
    rhousrms = mh2rms / volus
    # keep number
    beamrhoobs = rhousobs.to(u.Msun/u.Mpc**3)
    beamrhorms = rhousrms.to(u.Msun/u.Mpc**3)
    beamrholim = rhous.to(u.Msun/u.Mpc**3)

    obsunitdict = {'L': beamvalobs, 'dL': beamrmsobs,
                   'rho': beamrhoobs, 'drho': beamrhorms,
                   'nuobs_mean': meannuobs, 'z_mean': meanz}

    return obsunitdict

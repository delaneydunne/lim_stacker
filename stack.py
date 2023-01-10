from __future__ import absolute_import, print_function
from .tools import *
from .plottools import *
from .cubefilters import *
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
    # if np.any(np.isnan(pixval)):
    #     return None

    # check how many center aperture pixels are masked
    if np.sum(np.isnan(pixval).flatten()) > (params.freqwidth*params.xwidth**2)/2:
        return None

    # less than half of EACH SPECTRAL CHANNEL masked
    for i in range(pixval.shape[0]):
        if np.sum(np.isnan(pixval[i,:,:]).flatten()) > params.xwidth**2 / 2:
            return None

    # subtract global spectral mean
    if params.specmeanfilter:
        cutout = remove_cutout_spectral_mean(cutout, params)

    # check if the cutout failed the tests in these functions
    if not cutout:
        return None

    # subtract the per-channel means
    if params.chanmeanfilter:
        cutout = remove_cutout_chanmean(cutout, params)

    # check if the cutout failed the tests in these functions
    if not cutout:
        return None

    # subtract the low-order modes
    if params.lowmodefilter:
        cutout = remove_cutout_lowmodes(cutout, params)

    # check if the cutout failed the tests in these functions
    if not cutout:
        return None

    Tbval = np.nansum(pixval)
    Tbrms = np.sqrt(np.nansum(rmsval**2))

    # find the actual Tb in the cutout -- weighted average over all axes
    # Tbval, Tbrms = weightmean(pixval, rmsval)
    if np.isnan(Tbval):
        return None

    cutout.T = Tbval
    cutout.rms = Tbrms

    if params.obsunits:
        observer_units_weightedsum(pixval, rmsval, cutout, params)

    return cutout


""" OBSERVER UNIT FUNCTIONS """

def rayleigh_jeans(tb, nu, omega):
    """
    Rayleigh-Jeans law for conversion between brightness temperature and flux. Explicit
    version of u.brightness_temperature from astropy.units.equivalencies.
    -------
    INPUTS:
    -------
    tb:    brightness temperature in temperature units (should be a quantity)
    nu:    observed frequency in frequency units (should be a quantity)
    omega: beam solid angle convolved with solid angle of source. has to be in steradian
    --------
    OUTPUTS:
    --------
    jy:    specific flux associated with tb (will be a quantity with units of Jy)
    """
    jy_per_sr = 2*nu**2*const.k_B*tb / const.c**2

    jy = (jy_per_sr * omega.value).to(u.Jy)

    return jy

def line_luminosity(flux, nuobs, params, summed=True):
    """
    Function to calculate the (specifically CO) line luminosity of a line emitter from
    its flux. from Solomon et al. 1997 (https://iopscience.iop.org/article/10.1086/303765/pdf)
    -------
    INPUTS:
    -------
    flux:   brightness of the source in Jy (should be a quantity)
    nuobs:  central observed frequency in GHz (unitless float)
    params: lim_stacker params object (only used for central frequency)
    --------
    OUTPUTS:
    --------
    linelum: L'_CO in K km/s pc^2
    """

    dnuobs = 0.03125*u.GHz * (np.ones(len(flux)) - len(flux)//2)
    nuobs = nuobs*u.GHz + dnuobs

    # find redshift from nuobs:
    zval = freq_to_z(params.centfreq*u.GHz, nuobs)

    # luminosity distance in Mpc
    DLs = cosmo.luminosity_distance(zval)

    # line luminosity
    linelum = const.c**2 / (2*const.k_B) * flux * DLs**2 / (nuobs**2 * (1+zval)**3)

    # fix units
    linelum = linelum.to(u.K*u.km/u.s*u.pc**2)

    # if summed, sum across channels for an overall line luminosity
    if summed:
        linelum = np.sum(linelum)

    return linelum

def rho_h2(linelum, nuobs, params):
    """
    Function to calculate the (specifically CO) line luminosity of a line emitter from
    its flux. from Solomon et al. 1997 (https://iopscience.iop.org/article/10.1086/303765/pdf)
    -------
    INPUTS:
    -------
    linelum: line luminosity of the source in K km/s pc^2 (should be a quantity)
    nuobs:   observed frequency in frequency units (should be a quantity)
    params:  lim_stacker params object (used for central frequency and the size of the aperture,
             in order to calculate the cosmic volume covered by the aperture)
    --------
    OUTPUTS:
    --------
    rhoh2: molecular gas density in the aperture (Msun / Mpc^3; astropy quantity)
    """


    alphaco = 3.6*u.Msun / (u.K * u.km/u.s*u.pc**2)

    # h2 masses
    mh2 = linelum * alphaco

    nu1 = ((nuobs*u.GHz - params.freqwidth/2*0.03125*u.GHz).to(u.GHz)).value
    nu2 = ((nuobs*u.GHz + params.freqwidth/2*0.03125*u.GHz).to(u.GHz)).value

    (z, z1, z2) = freq_to_z(params.centfreq, np.array([nuobs, nu1, nu2]))

    distdiff = cosmo.luminosity_distance(z1) - cosmo.luminosity_distance(z2)

    # proper volume of the cube
    # volus = ((cosmo.kpc_proper_per_arcmin(z) * params.xwidth * 2*u.arcmin).to(u.Mpc))**2 * distdiff
    beamx = 4.5*u.arcmin/(2*np.sqrt(2*np.log(2)))
    volus = ((cosmo.kpc_proper_per_arcmin(z) * params.xwidth * beamx).to(u.Mpc))**2 * distdiff

    rhoh2 = (mh2 / volus).to(u.Msun/u.Mpc**3)

    return rhoh2


def perchannel_flux_sum(tbvals, rmsvals, nuobs, params):
    """
    Function to determine the per-channel flux (in Jy km/s) of a cutout from a COMAP map.
    This will take the UNWEIGHTED SUM of all the spaxel values in each channel. If there
    are nans in a channel, it fill them in by interpolating them to be the mean value in
    the channel.
    -------
    INPUTS:
    -------
    tbvals:  array of brightness temperature values. first axis needs to be the spectral one
             should be unitless (NxMxL)
    rmsvals: the per-pixel rms uncertainties associated with tbvals. also a unitless array (NxMxL)
    nuobs:   observed frequency in frequency units (should be a float, in GHz)
    params:  lim_stacker params object (only used for central frequency)
    --------
    OUTPUTS:
    --------
    Sval_chans: the flux in the cutout in each individual spectral channel. length-N array of astropy
                quantities (in Jy km/s)
    Srms_chans: the rms associated with each flux value. length-N array of astropy quantities (in Jy km/s)
    """

    # number of frequency values we're dealing with
    freqwidth = tbvals.shape[0]

    # taking a straight sum, so not including certain voxels because they're NaNed
    # out will cause problems. interpolate to fill them (bad)
    tbvals, rmsvals = cubelet_fill_nans(tbvals, rmsvals, params)

    # correct for the primary beam response
    tbvals = tbvals / 0.72
    rmsvals = rmsvals / 0.72

    # not the COMAP beam but the angular size of the region over which the brightness
    # temperature is the given value (ie one spaxel)
    omega_B = ((2*u.arcmin)**2).to(u.sr)

    # central frequency of each individual spectral channel
    nuobsvals = (np.arange(params.freqwidth) - params.freqwidth//2) * 31.25*u.MHz
    nuobsvals = nuobsvals + nuobs*u.GHz

    Sval_chans = np.ones(params.freqwidth)*u.Jy
    Srms_chans = np.ones(params.freqwidth)*u.Jy
    # flux in each spectral channel
    for i in range(params.freqwidth):
        Sval_chan = rayleigh_jeans(tbvals[i,:,:]*u.K, nuobsvals[i], omega_B)
        Sval_chan = np.nansum(Sval_chan)

        Srms_chan = rayleigh_jeans(rmsvals[i,:,:]*u.K, nuobsvals[i], omega_B)
        Srms_chan = np.sqrt(np.nansum(Srms_chan**2))

        Sval_chans[i] = Sval_chan
        Srms_chans[i] = Srms_chan

    # channel widths in km/s
    delnus = (31.25*u.MHz / nuobsvals * const.c).to(u.km/u.s)

    Snu_Delnu = Sval_chans * delnus
    d_Snu_Delnu = Srms_chans * delnus

    return Snu_Delnu, d_Snu_Delnu

def perchannel_flux_mean(tbvals, rmsvals, nuobs, params):
    """
    Function to determine the per-channel flux (in Jy km/s) of a cutout from a COMAP map.
    This will INVERSE-VARIANCE weight each spaxel by its associated RMS, and thus will ignore
    NaNs.
    -------
    INPUTS:
    -------
    tbvals:  array of brightness temperature values. first axis needs to be the spectral one
             should be unitless (NxMxL)
    rmsvals: the per-pixel rms uncertainties associated with tbvals. also a unitless array (NxMxL)
    nuobs:   observed frequency in frequency units (should be a float, in GHz)
    params:  lim_stacker params object (only used for central frequency)
    --------
    OUTPUTS:
    --------
    Sval_chans: the flux in the cutout in each individual spectral channel. length-N array of astropy
                quantities (in Jy km/s)
    Srms_chans: the rms associated with each flux value. length-N array of astropy quantities (in Jy km/s)
    """

    # number of frequency channels we're dealing with
    freqwidth = tbvals.shape[0]

    # correct for the primary beam response
    tbvals = tbvals / 0.72
    rmsvals = rmsvals / 0.72

    # not the COMAP beam but the angular size of the region over which the brightness
    # temperature is the given value (ie one spaxel)
    # omega_B = ((params.xwidth * 2*u.arcmin)**2).to(u.sr)

    # actual COMAP beam
    beam_fwhm = 4.5*u.arcmin
    sigma_x = beam_fwhm / (2 * np.sqrt(2 * np.log(2)))
    sigma_y = sigma_x
    omega_B = (2 * np.pi * sigma_x * sigma_y).to(u.sr)

    # central frequency of each individual spectral channel
    nuobsvals = (np.arange(freqwidth) - params.freqwidth//2) * 31.25*u.MHz
    nuobsvals = nuobsvals + nuobs*u.GHz

    Sval_chans = np.ones(freqwidth)*u.Jy
    Srms_chans = np.ones(freqwidth)*u.Jy
    # flux in each spectral channel
    for i in range(freqwidth):
        Sval_chan = rayleigh_jeans(tbvals[i,:,:]*u.K, nuobsvals[i], omega_B)

        Srms_chan = rayleigh_jeans(rmsvals[i,:,:]*u.K, nuobsvals[i], omega_B)

        # per-channel flux density by taking the weighted mean
        Sval_chan, Srms_chan = weightmean(Sval_chan, Srms_chan)

        Sval_chans[i] = Sval_chan
        Srms_chans[i] = Srms_chan

    # channel widths in km/s
    delnus = (31.25*u.MHz / nuobsvals * const.c).to(u.km/u.s)

    Snu_Delnu = Sval_chans * delnus
    d_Snu_Delnu = Srms_chans * delnus

    return Snu_Delnu, d_Snu_Delnu


def observer_units_sum(tbvals, rmsvals, cutout, params):
    """
    calculate the more physical quantities associated with a single cutout. Uses
    an UNWEIGHTED SUM to get the per-channel flux, interpolating across NaNs.
    """

    # per-channel fluxes
    Sval_chan, Srms_chan = perchannel_flux_sum(tbvals, rmsvals, cutout.freq, params)

    # make the fluxes into line luminosities
    linelum = line_luminosity(Sval_chan, cutout.freq, params)
    linelumrms = line_luminosity(Srms_chan, cutout.freq, params)

    rhoh2 = rho_h2(linelum, cutout.freq, params)
    rhoh2rms = rho_h2(linelumrms, cutout.freq, params)

    cutout.flux = Sval_chan
    cutout.dflux = Srms_chan

    cutout.linelum = linelum
    cutout.dlinelum = linelumrms

    cutout.rhoh2 = rhoh2
    cutout.drhoh2 = rhoh2rms

    return cutout


def observer_units_weightedsum(tbvals, rmsvals, cutout, params):
    """
    calculate the more physical quantities associated with a single cutout. Uses
    a WEIGHTED SUM to get the per-channel flux, and thus ignores NaNs.
    """

    # per-channel fluxes
    Sval_chan, Srms_chan = perchannel_flux_mean(tbvals, rmsvals, cutout.freq, params)

    # make the fluxes into line luminosities
    linelum = line_luminosity(Sval_chan, cutout.freq, params)
    linelumrms = line_luminosity(Srms_chan, cutout.freq, params)

    rhoh2 = rho_h2(linelum, cutout.freq, params)
    rhoh2rms = rho_h2(linelumrms, cutout.freq, params)

    cutout.flux = Sval_chan.value
    cutout.dflux = Srms_chan.value

    cutout.linelum = linelum.value
    cutout.dlinelum = linelumrms.value

    cutout.rhoh2 = rhoh2.value
    cutout.drhoh2 = rhoh2rms.value

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

def aperture_collapse_cubelet_space(cutout, params, linelum=False):
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
    if not linelum:
        freqstack = np.nansum(fpixval, axis=(1,2))
        rmsfreqstack = np.sqrt(np.nansum(frmsval**2, axis=(1,2)))
        # freqstack, rmsfreqstack = weightmean(fpixval, frmsval, axis=(1,2))

        cutout.freqstack = freqstack
        cutout.freqstackrms = rmsfreqstack

    else:
        # calculate the per-channel line luminosity in the spatial apertures and
        # get the spectrum from those values
        flux, dflux = perchannel_flux_mean(fpixval, frmsval, cutout.freq, params)
        linelum = line_luminosity(flux, cutout.freq, params, summed=False)
        dlinelum = line_luminosity(dflux, cutout.freq, params, summed=False)

        cutout.freqstack = linelum.value
        cutout.freqstackrms = dlinelum.value

    return

def cubelet_fill_nans(pixvals, rmsvals, params):
    """
    function to replace any NaN values in a cubelet aperture with the mean in that
    frequency channel. should obviously check to make sure there are a reasonable
    number of actual values first.
    """

    for i in range(pixvals.shape[0]):
        chanmean, chanrms = weightmean(pixvals[i,:,:], rmsvals[i,:,:])
        pixvals[i, np.where(np.isnan(pixvals[i,:,:]))] = chanmean
        rmsvals[i, np.where(np.isnan(rmsvals[i,:,:]))] = chanrms

    return pixvals, rmsvals


""" ACTUAL STACKING """

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

            # brightness temperature only cubelets
            if params.cubelet:
                aperture_collapse_cubelet_freq(cutout, params)
                aperture_collapse_cubelet_space(cutout, params, linelum=params.obsunits)

                if ti == 0:
                    cubespec, cubespecrms = cutout.freqstack, cutout.freqstackrms
                    cubeim, cubeimrms = cutout.spacestack, cutout.spacestackrms
                    ti = 1
                else:
                    scspec = np.stack((cubespec, cutout.freqstack))
                    scspecrms = np.stack((cubespecrms, cutout.freqstackrms))
                    scim = np.stack((cubeim, cutout.spacestack))
                    scimrms = np.stack((cubeimrms, cutout.spacestackrms))

                    cubespec, cubespecrms = weightmean(scspec, scspecrms, axis=0)
                    cubeim, cubeimrms = weightmean(scim, scimrms, axis=0)

                # delete the other arrays
                cutout.__delattr__('cubestack')
                cutout.__delattr__('cubestackrms')
                cutout.__delattr__('freqstack')
                cutout.__delattr__('freqstackrms')
                cutout.__delattr__('spacestack')
                cutout.__delattr__('spacestackrms')

            cutoutlist.append(cutout)
            if goalnobj:
                field_nobj += 1

                if field_nobj == goalnobj:
                    if params.verbose:
                        print("Hit goal number of {} cutouts".format(goalnobj))
                    if params.cubelet:
                        return cutoutlist, [cubestack, cuberms]
                    else:
                        return cutoutlist

        if params.verbose:
            if i % 100 == 0:
                print('   done {} of {} cutouts in this field'.format(i, galcat.nobj))

    if params.cubelet:
        return cutoutlist, [cubeim, cubeimrms, cubespec, cubespecrms]
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
        stackim = []
        imrms = []
        stackspec = []
        specrms = []
    for i in range(len(maplist)):
        if params.cubelet:
            fieldcutouts, fieldstacks = field_get_cutouts(maplist[i], galcatlist[i],
                                                          params, field=fields[i],
                                                          goalnobj=numcutoutlist[i])
            if isinstance(fieldstacks[0], np.ndarray):
                stackim.append(fieldstacks[0])
                imrms.append(fieldstacks[1])
                stackspec.append(fieldstacks[2])
                specrms.append(fieldstacks[3])

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
        stackim, imrms = np.stack(stackim), np.stack(imrms)
        stackspec, specrms = np.stack(stackspec), np.stack(specrms)

        stackim, imrms = weightmean(stackim, imrms, axis=0)
        stackspec, specrms = weightmean(stackspec, specrms, axis=0)
        print(stackspec)

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
        # allou = observer_units(cutlistdict['T'], cutlistdict['rms'], cutlistdict['z'],
        #                        cutlistdict['freq'], params)

        allou = {'L': cutlistdict['linelum'], 'dL': cutlistdict['dlinelum'],
                 'rho': cutlistdict['rhoh2'], 'drho': cutlistdict['drhoh2']}

        linelumstack, dlinelumstack = weightmean(allou['L'], allou['dL'])
        rhoh2stack, drhoh2stack = weightmean(allou['rho'], allou['drho'])

        outputvals['linelum'], outputvals['dlinelum'] = linelumstack, dlinelumstack
        outputvals['rhoh2'], outputvals['drhoh2'] = rhoh2stack, drhoh2stack
        outputvals['nuobs_mean'], outputvals['z_mean'] = np.nanmean(cutlistdict['freq']), np.nanmean(cutlistdict['z'])

        print(outputvals)

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
        if not params.cubelet:
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

    # if params.plotcubelet:
    #     cubelet_plotter(cubestack, cuberms, params)

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

        # if params.cubelet:
        #     cubefile = params.datasavepath + '/stacked_3d_cubelet.npz'
        #     np.savez(cubefile, T=cubestack, rms=cuberms)

    # objects to be returned
    if params.cubelet:
        returns = [outputvals, stackim, stackspec, fieldcatidx]#, cubestack, cuberms]
    else:
        returns = [outputvals, stackim, stackspec, fieldcatidx]

    # return list of all cutouts w associated metadata as well if asked to
    if params.returncutlist:
        returns.append(allcutouts)

    return returns

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

    # objects to be returned
    if params.cubelet:
        returns = [outputvals, stackim, stackspec, fieldcatidx, cubestack, cuberms]
    else:
        returns = [outputvals, stackim, stackspec, fieldcatidx]

    # return list of all cutouts w associated metadata as well if asked to
    if params.returncutlist:
        returns.append(allcutouts)

    return returns


def observer_units(Tvals, rmsvals, zvals, nuobsvals, params):
    """
    unit change to physical units
    """

    # main beam to full beam correction
    Tvals = Tvals / 0.72
    rmsvals = rmsvals / 0.72

    # actual beam FWHP is a function of frequency - listed values are 4.9,4.5,4.4 arcmin at 26, 30, 34GHz
    # set up a function to interpolate on
    # beamthetavals = np.array([4.9,4.5,4.4])
    # beamthetafreqs = np.array([26, 30, 34])

    # beamthetas = np.interp(nuobsvals, beamthetafreqs, beamthetavals)*u.arcmin
    # omega_Bs = 1.33*beamthetas**2

    # the 'beam' here is actually the stack aperture size
    # beamsigma = params.xwidth / 2 * 2*u.arcmin
    # omega_B = (2 / np.sqrt(2*np.log(2)))*np.pi*beamsigma**2
    omega_B = (2*u.arcmin)**2

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

    nu1 = nuobsvals - params.freqwidth/2*0.03125*u.GHz
    nu2 = nuobsvals + params.freqwidth/2*0.03125*u.GHz

    z = (params.centfreq*u.GHz - nuobsvals) / nuobsvals
    z1 = (params.centfreq*u.GHz - nu1) / nu1
    z2 = (params.centfreq*u.GHz - nu2) / nu2
    meanz = np.nanmean(z)

    distdiff = cosmo.luminosity_distance(z1) - cosmo.luminosity_distance(z2)

    # proper volume of the cube
    volus = ((cosmo.kpc_proper_per_arcmin(z1) * params.xwidth * 2*u.arcmin).to(u.Mpc))**2 * distdiff

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

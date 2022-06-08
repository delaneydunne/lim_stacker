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
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", message="invalid value encountered in power")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")

# standard COMAP cosmology
cosmo = FlatLambdaCDM(H0=70*u.km / (u.Mpc*u.s), Om0=0.286, Ob0=0.047)

def single_cutout(idx, galcat, comap, params):

    # find gal in each axis, test to make sure it falls into field
    ## freq
    zval = galcat.z[idx]
    nuobs = params.centfreq / (1 + zval)
    if nuobs < comap.flims[0] or nuobs > comap.flims[-1]:
        return None
    freqidx = np.max(np.where(comap.freq < nuobs))
    if np.abs(nuobs - comap.freq[freqidx]) < comap.fstep / 2:
        fdiff = -1
    else:
        fdiff = 1

    x = galcat.coords[idx].ra.deg
    if x < comap.xlims[0] or x > comap.xlims[-1]:
        return None
    xidx = np.max(np.where(comap.ra < x))
    if np.abs(x - comap.ra[xidx]) < comap.xstep / 2:
        xdiff = -1
    else:
        xdiff = 1

    y = galcat.coords[idx].dec.deg
    if y < comap.ylims[0] or y > comap.ylims[-1]:
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

def single_cutout_old(idx, galcat, comap, params):

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

        if params.verbose:
            if i % 100 == 0:
                print('   done {} of {} cutouts in this field'.format(i, galcat.nobj))

    return cutoutlist

def stacker(maplist, galcatlist, params, cmap='PiYG_r'):
    """
    wrapper to perform a full stack on all available values in the catalogue.
    will plot if desired
    """
    fields = [1,2,3]
    # dict to store stacked values
    outputvals = {}

    fieldlens = []
    allcutouts = []
    for i in range(len(maplist)):
        fieldcutouts = field_get_cutouts(maplist[i], galcatlist[i], params, field=fields[i])
        fieldlens.append(len(fieldcutouts))
        allcutouts = allcutouts + fieldcutouts

        if params.verbose:
            print('Field {} complete'.format(fields[i]))

    print(fieldlens)

    # unzip all your cutout objects
    Tvals = []
    rmsvals = []
    zvals = []
    nuobsvals = []
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
        zvals.append(cut.z)
        nuobsvals.append(cut.freq)
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
    zvals = np.array(zvals)
    nuobsvals = np.array(nuobsvals)
    catidxs = np.array(catidxs)
    if params.spacestackwidth:
        spacestack = np.array(spacestack)
        spacerms = np.array(spacerms)
    if params.freqstackwidth:
        freqstack = np.array(freqstack)
        freqrms = np.array(freqrms)

    # put into physical units if requested
    if params.obsunits:
        allou = observer_units(Tvals, rmsvals, zvals, nuobsvals, params)


        linelumstack, dlinelumstack = weightmean(allou['L'], allou['dL'])
        rhoh2stack, drhoh2stack = weightmean(allou['rho'], allou['drho'])

        outputvals['linelum'], outputvals['dlinelum'] = linelumstack, dlinelumstack
        outputvals['rhoh2'], outputvals['drhoh2'] = rhoh2stack, drhoh2stack

    # split indices up by field for easy access later
    fieldcatidx = []
    previdx = 0
    for catlen in fieldlens:
        fieldcatidx.append(catidxs[previdx:catlen+previdx])
        previdx += catlen

    # overall stack for T value
    stacktemp, stackrms = weightmean(Tvals, rmsvals)
    outputvals['T'], outputvals['rms'] = stacktemp, stackrms

    """ EXTRA STACKS """
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

    """ PLOTS """
    if params.saveplots:
        # make the directory to store the plots
        os.makedirs(params.savepath, exist_ok=True)

    if params.plotspace:
        spatial_plotter(stackim, params, cmap=cmap)

    if params.plotfreq:
        spectral_plotter(stackspec, params)

    if params.plotspace and params.plotfreq:
        combined_plotter(stackim, stackspec, params, cmap=cmap, stackresult=(stacktemp*1e6,stackrms*1e6))

    return outputvals, stackim, stackspec, fieldcatidx

def observer_units(Tvals, rmsvals, zvals, nuobsvals, params):
    """
    unit change to physical units
    """

    # actual beam FWHP is a function of frequency - listed values are 4.9,4.5,4.4 arcmin at 26, 30, 34GHz
    # set up a function to interpolate on
    beamthetavals = np.array([4.9,4.5,4.4])
    beamthetafreqs = np.array([26, 30, 34])

    beamsigma = 2*u.arcmin
    omega_B = (2 / np.sqrt(2*np.log(2)))*np.pi*beamsigma**2

    beamthetas = np.interp(nuobsvals, beamthetafreqs, beamthetavals)*u.arcmin
    omega_Bs = 1.33*beamthetas**2

    nuobsvals = nuobsvals*u.GHz

    onesiglimvals = Tvals + rmsvals

    Sact = (Tvals*u.K).to(u.Jy, equivalencies=u.brightness_temperature(omega_B, nuobsvals))
    Ssig = (onesiglimvals*u.K).to(u.Jy, equivalencies=u.brightness_temperature(omega_B, nuobsvals))
    Srms = (rmsvals*u.K).to(u.Jy, equivalencies=u.brightness_temperature(omega_B, nuobsvals))

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

    nu1 = nuobsvals - 0.5*0.0625*u.GHz
    nu2 = nuobsvals + 0.5*0.0625*u.GHz

    z1 = (115.27*u.GHz - nu1) / nu1
    z2 = (115.27*u.GHz - nu2) / nu2

    distdiff = cosmo.luminosity_distance(z1) - cosmo.luminosity_distance(z2)

    # proper volume at each voxel
    volus = ((cosmo.kpc_proper_per_arcmin(z1) * 4*u.arcmin).to(u.Mpc))**2 * distdiff

    rhous = mh2us / volus
    rhousobs = mh2obs / volus
    rhousrms = mh2rms / volus
    # keep number
    beamrhoobs = rhousobs.to(u.Msun/u.Mpc**3)
    beamrhorms = rhousrms.to(u.Msun/u.Mpc**3)
    beamrholim = rhous.to(u.Msun/u.Mpc**3)

    obsunitdict = {'L': beamvalobs, 'dL': beamrmsobs,
                   'rho': beamrhoobs, 'drho': beamrhorms}

    return obsunitdict

""" PLOTTING FUNCTIONS """
def spatial_plotter(stackim, params, cmap='PiYG_r'):

    # corners for the beam rectangle
    if params.xwidth % 2 == 0:
        rectmin = params.spacestackwidth - params.xwidth/2 - 0.5
        rectmax = params.spacestackwidth + params.xwidth/2 - 0.5
    else:
        rectmin = params.spacestackwidth - params.xwidth/2
        rectmax = params.spacestackwidth + params.xwidth/2

    xcorners = (rectmin, rectmin, rectmax, rectmax, rectmin)
    ycorners = (rectmin, rectmax, rectmax, rectmin, rectmin)

    vext = np.nanmax(np.abs([np.nanmin(stackim*1e6), np.nanmax(stackim*1e6)]))
    vmin,vmax = -vext, vext

    # unsmoothed
    fig, ax = plt.subplots(1)
    c = ax.imshow(stackim*1e6, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.plot(xcorners, ycorners, color='0.8', linewidth=4, zorder=10)
    cbar = fig.colorbar(c)
    cbar.ax.set_ylabel('Tb (uK)')

    if params.saveplots:
        fig.savefig(params.savepath+'/angularstack_unsmoothed.png')

    # smoothed
    smoothed_spacestack_gauss = convolve(stackim, params.gauss_kernel)

    vext = np.nanmax(smoothed_spacestack_gauss*1e6)

    fig, ax = plt.subplots(1)
    c = ax.imshow(smoothed_spacestack_gauss*1e6, cmap=cmap, vmin=-vext, vmax=vext)
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
        fig.savefig(params.savepath + '/frequencystack.png')

    return 0

def combined_plotter(stackim, stackspec, params, cmap='PiYG_r', stackresult=None):

    # corners for the beam rectangle
    if params.xwidth % 2 == 0:
        rectmin = params.spacestackwidth - params.xwidth/2 - 0.5
        rectmax = params.spacestackwidth + params.xwidth/2 - 0.5
    else:
        rectmin = params.spacestackwidth - params.xwidth/2
        rectmax = params.spacestackwidth + params.xwidth/2

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

    c = axs[0,0].imshow(stackim*1e6, cmap=cmap, vmin=vmin, vmax=vmax)
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
    vext = np.nanmax(smoothed_spacestack_gauss*1e6)
    c = axs[0,1].imshow(smoothed_spacestack_gauss*1e6, cmap=cmap, vmin=-vext, vmax=vext)
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
        fig.savefig(params.savepath + '/combinedstackim.png')

    return 0

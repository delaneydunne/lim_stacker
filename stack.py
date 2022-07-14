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
import csv
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
        cutout.freqstack = freqstack
        cutout.freqstackrms = rmsfreqstack

    return cutout

def field_get_cutouts(comap, galcat, params, field=None):
    """
    wrapper to return all cutouts for a single field
    """
    ti = 0
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
                                                                           field=fields[i])
            if fieldcubestack:
                cubestacks.append(fieldcubestack)
                cubermss.append(fieldcuberms)
        else:
            fieldcutouts = field_get_cutouts(maplist[i], galcatlist[i], params, field=fields[i])
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
    if params.savedata or params.saveplots:
        make_output_pathnames(params)

    if params.saveplots:
        catalogue_plotter(galcatlist, fieldcatidx, params)

    if params.spacestackwidth and params.plotspace:
        spatial_plotter(stackim, params, cmap=cmap)

    if params.freqstackwidth and params.plotfreq:
        spectral_plotter(stackspec, params)

    if params.spacestackwidth and params.freqstackwidth and params.plotspace and params.plotfreq:
        combined_plotter(stackim, stackspec, params, cmap=cmap, stackresult=(stacktemp*1e6,stackrms*1e6))

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

    # actual beam FWHP is a function of frequency - listed values are 4.9,4.5,4.4 arcmin at 26, 30, 34GHz
    # set up a function to interpolate on
    beamthetavals = np.array([4.9,4.5,4.4])
    beamthetafreqs = np.array([26, 30, 34])

    beamsigma = 2*u.arcmin
    omega_B = (2 / np.sqrt(2*np.log(2)))*np.pi*beamsigma**2

    beamthetas = np.interp(nuobsvals, beamthetafreqs, beamthetavals)*u.arcmin
    omega_Bs = 1.33*beamthetas**2

    nuobsvals = nuobsvals*u.GHz
    meannuobs = np.nanmean(nuobsvals)

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

    z = (params.centfreq*u.GHz - nuobsvals) / nuobsvals
    z1 = (params.centfreq*u.GHz - nu1) / nu1
    z2 = (params.centfreq*u.GHz - nu2) / nu2
    meanz = np.nanmean(z)

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
                   'rho': beamrhoobs, 'drho': beamrhorms,
                   'nuobs_mean': meannuobs, 'z_mean': meanz}

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
        fig.savefig(params.plotsavepath+'/angularstack_unsmoothed.png')

    # smoothed
    smoothed_spacestack_gauss = convolve(stackim, params.gauss_kernel)

    vext = np.nanmax(smoothed_spacestack_gauss*1e6)

    fig, ax = plt.subplots(1)
    c = ax.imshow(smoothed_spacestack_gauss*1e6, cmap=cmap, vmin=-vext, vmax=vext)
    ax.plot(xcorners, ycorners, color='0.8', linewidth=4, zorder=10)
    cbar = fig.colorbar(c)
    cbar.ax.set_ylabel('Tb (uK)')

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

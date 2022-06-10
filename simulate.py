from __future__ import absolute_import, print_function
from .tools import *
from .stack import *
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import os
import pandas as pd
import h5py
import glob
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.io import fits
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", message="invalid value encountered in power")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")


""" COMBINE ALREADY-MADE SIMS WITH DATA """
def load_sim(file):
    """
    loads in a mock CO luminosity cube and stores as an object like the real
    map
    """

    simmap = empty_table()

    with np.load(file) as simfile:
        # sims output uK, data in K. stack functions all deal w K so convert
        simmap.rawmap = simfile['map_cube'].T / 1e6
        simmap.freq = simfile['map_frequencies']
        simmap.ra = simfile['map_pixel_ra']
        simmap.dec = simfile['map_pixel_dec']

    # **anything else here?

    return simmap

def beam_smooth_sim(simmap, fwhm=4.5):
    """
    Smooth the simulated data with a Gaussian 2D kernel to approximate the wider COMAP beam
    fwhm should be the beam fwhm in arcmin
    """

    std = fwhm / (2*np.sqrt(2*np.log(2)))
    pixwidth = (simmap.ra[1] - simmap.ra[0])*60

    std_pix = std / pixwidth

    beamkernel = Gaussian2DKernel(std_pix)

    # loop over the frequency axis and convolve each frame
    smoothsimlist = []
    for i in range(len(simmap.freq)):
        smoothsimlist.append(convolve(simmap.rawmap[i,:,:], beamkernel))

    simmap.map = np.stack(smoothsimlist, axis=0)

    return simmap

def dump_map(comap, filename):
    """
    save a map class as a hdf5 file
    this will be more bare-bones than the actual datafiles -- just including the
    datasets needed for stacking
    """

    # *** automatic output file naming

    # undo the coordinate shift so it doesn't happen twice when it's reloaded
    comap.freq = comap.freq + comap.fstep / 2
    comap.ra = comap.ra + comap.xstep / 2
    comap.dec = comap.dec + comap.ystep / 2

    with h5py.File(filename, 'w') as f:
        dset = f.create_dataset('map_coadd', data = comap.simdatmap, dtype='float64')
        dset = f.create_dataset('rms_coadd', data = comap.rms, dtype='float64')
        dset = f.create_dataset('freq', data = comap.freq, dtype='float64')
        dset = f.create_dataset('x', data = comap.ra, dtype='float64')
        dset = f.create_dataset('y', data = comap.dec, dtype='float64')

        patchcent = (comap.fieldcent.ra.deg, comap.fieldcent.dec.deg)
        dset = f.create_dataset('patch_center', data = patchcent, dtype='float64')

    return 0

def scalesim(datfiles, simfiles, outfiles, scale=1, beamfwhm=4.5, save=True,
             rmsscale=False):
    """
    Wrapper to load files and add simulated data. Scale can be arraylike or a single value
    ***warn properly
    """

    # *** fix
    # if len(datfiles) != len(simfiles):
    #     print('different number of files')
    #     return 0

    # if an array of scale values passed, only run io once and then add and dump
    #  for each scale value
    if isinstance(scale, (list, tuple, np.ndarray)):
        for i in range(len(datfiles)):

            datmap = load_map(datfiles[i])

            simlummap = load_sim(simfiles[i])
            simlummap = beam_smooth_sim(simlummap, fwhm=beamfwhm)

            datmap.simmap = np.array(simlummap.map)

            if i == 0:
                # noise varies between the comap fields -- only want to scale
                # in field 1 and have the others match
                if rmsscale:
                    # *** generalize
                    tm = datmap.map[100,40:80,40:80]
                    maprms = np.abs(np.nanmedian(tm))

                    tsm = datmap.simmap[100,40:80,40:80]
                    simsig = np.nanmax(tsm) * 0.25

                    rawsn = simsig / maprms

                    scale = scale * rawsn
                    # ***return better
                    print(scale)

            for j in range(len(scale)):
                simdatmap = np.array(datmap.map / scale[j] + datmap.simmap)
                sdmcent = simdatmap[120:160,40:80,40:80]

                # subtract the mean
                meanval = np.nanmean(sdmcent)
                datmap.simdatmap = simdatmap - meanval

                if rmsscale:
                    # rename the output files to have the correct scale in them
                    outfiles[i] = outfiles[i].split('_sn')[0] + '_sn'+str(j)+'.h5'
                else:
                    # rename the output files to have the correct scale in them
                    outfiles[i] = outfiles[i].split('_scale')[0] + '_scale'+str(j)+'.h5'

                if save:
                    dump_map(datmap, outfiles[i])

    # otherwise, just run through everything once
    else:
        for i in range(len(datfiles)):

            datmap = load_map(datfiles[i])

            simlummap = load_sim(simfiles[i])
            simlummap = beam_smooth_sim(simlummap, fwhm=beamfwhm)

            datmap.simmap = np.array(simlummap.map)

            if i == 0:
                # noise varies between the comap fields -- only want to scale
                # in field 1 and have the others match
                if rmsscale:
                    # *** generalize
                    tm = datmap.map[120:160,40:80,40:80]
                    maprms = np.abs(np.nanmean(tm))

                    tsm = datmap.simmap[120:160,40:80,40:80]
                    simsig = np.nanmax(tsm) * 0.25

                    rawsn = simsig / maprms

                    scale = scale * rawsn
                    # ***return better
                    print(scale)

            simdatmap = np.array(datmap.map / scale + datmap.simmap)
            sdmcent = simdatmap[120:160,40:80,40:80]

            # subtract off the mean (done in the actual COMAP pipeline)
            meanval = np.nanmean(sdmcent)
            datmap.simdatmap = simdatmap - meanval

            if save:
                dump_map(datmap, outfiles[i])

    return datmap


""" DISTRIBUTION-AWARE STACKS """
def bin_field_sim_catalogue(actidxs, galcat, simcat, params):
    """
    bin the simulated galaxy catalogue to match the real one in redshift
    """
    nperbin, zedges = field_zbin_stack_output(actidxs, 0, galcat, params)
    simz = simcat.z
    coords = simcat.coords
    idx = simcat.idx

    bincatlist = []
    for i in range(params.nzbins):
        binhaloidx = np.where(np.logical_and(simz > zedges[i], simz < zedges[i+1]))[0]

        bincat = empty_table()

        bincat.z = simz[binhaloidx]
        bincat.coords = coords[binhaloidx]
        bincat.idx = idx[binhaloidx]
        bincat.goalnobj = nperbin[i]
        bincat.nobj = len(binhaloidx)
        bincatlist.append(bincat)

        if bincat.goalnobj > bincat.nobj:
            print("Too few objects in simulated catalogue!")
            return None

    return nperbin, bincatlist

def bin_get_sim_cutouts(comap, galcat, params, field=None):
    """
    wrapper to return all cutouts for a single bin
    """

    cutoutlist = []
    ngood = 0
    for i in range(galcat.nobj):
        cutout = single_cutout(i, galcat, comap, params)

        # if it passed all the tests, keep it
        if cutout:
            if field:
                cutout.field = field

            cutoutlist.append(cutout)

            ngood += 1

            if ngood == galcat.goalnobj:
                return cutoutlist


        if params.verbose:
            if i % 100 == 0:
                print('   done {} of {} cutouts in this field'.format(i, galcat.nobj))

    return None

def field_get_sim_cutouts(actidxs, comap, galcat, simcat, params, field=None, verbose=False):
    """
    return the appropriate number of simulated cutouts, binned in redshift to match galidxs
    """

    nperbin, bincatlist = bin_field_sim_catalogue(actidxs, galcat, simcat, params)

    cutoutlist = []
    for i in range(params.nzbins):
        bincat = bincatlist[i]

        if verbose:
            print("  bin {} needs {} cutouts".format(i+1, bincat.goalnobj))

        gbinlist = bin_get_sim_cutouts(comap, bincat, params)

        if gbinlist:
            cutoutlist = cutoutlist + gbinlist
        else:
            print("Didn't get enough gals in {} bin".format(i))
            break
    return cutoutlist

def sim_stacker(actcatidx, maplist, galcatlist, simcatlist, params):
    """
    wrapper to perform a stack on random locations binned to match
    the numbers of the stack in actcatidx
    """

    fields = [1,2,3]
    fieldlens = [len(actcatidx[0]), len(actcatidx[1]), len(actcatidx[2])]

    allcutouts = []
    for i in range(len(maplist)):
        if params.verbose:
            print(fields[i])
            print("need {} total cutouts".format(fieldlens[i]))
        fieldcutouts = field_get_sim_cutouts(actcatidx[i], maplist[i],
                                              galcatlist[i], simcatlist[i], params,
                                              field=fields[i], verbose=params.verbose)
        allcutouts = allcutouts + fieldcutouts

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
    stackvals = {'T':stacktemp, 'rms':stackrms}

    # overall spatial stack
    if params.spacestackwidth:
        stackim, imrms = weightmean(spacestack, spacerms, axis=0)
    else:
        stackspec, imrms = None, None

    # overall frequency stack
    if params.freqstackwidth:
        stackspec, specrms = weightmean(freqstack, freqrms, axis=0)
    else:
        stackim, imrms = None, None

    if params.saveplots:
        # make the directory to store the plots
        os.makedirs(params.savepath, exist_ok=True)

    if params.plotspace:
        spatial_plotter(stackim, params)

    if params.plotfreq:
        spectral_plotter(stackspec, params)

    if params.plotspace and params.plotfreq:
        combined_plotter(stackim, stackspec, params)

    return stackvals, stackim, stackspec, fieldcatidx

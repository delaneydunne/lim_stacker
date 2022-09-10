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
from astropy.convolution import convolve_fft, Gaussian1DKernel, Kernel
from astropy.io import fits
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", message="invalid value encountered in power")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")


""" CATALOGUE SELECTION FUNCTIONS """
def random_mass_subset(cat, params, seed=12345, massbins=3, in_place=True):
    """
    Function to make a more realistic selection of haloes from the full simulation
    peak-patch catalogue -- it will split the haloes into 'massbins' number of
    bins, and then pull params.goalnumcutouts haloes randomly from the top mass
    bin.
    """

    # how many objects are in the top mass bin
    bincutoff = cat.nobj // massbins

    if in_place:
        # sort the catalogue on mass (in descending order)
        try:
            cat.sort('M')
        except AttributeError:
            warnings.warn("There aren't any mass values in the input catalogue", RuntimeWarning)
            return
        # clip out the top mass bin
        cat.subset(np.arange(bincutoff))

    else:
        # make a copy of the catalogue and sort it on mass
        cutcat = cat.copy()
        try:
            cutcat.sort('M')
        except AttributeError:
            warnings.warn("There aren't any mass values in the input catalogue", RuntimeWarning)
            return
        # clip out the top mass bin
        cutcat.subset(np.arange(bincutoff))

    # random number generator
    rng = np.random.default_rng(seed)

    # shuffle the top mass bin -- the code will pull indices in order, so this will
    # mean a random subset of the top mass bin is used to get the goal number of objects
    indices = np.arange(bincutoff)
    rng.shuffle(indices)

    if in_place:
        cat.subset(indices)
        return
    else:
        cutcat.subset(indices)
        return cutcat




""" SETUP FUNCTIONS """
def sim_field_setup(pipemapfile, catfile, params, rawsimfile=None, outcatfile=None):
    """
    wrapper function to load in data (and match its WCS) for a simulated stack run
    """

    # load in main map and main catalogue
    pipemap = maps(pipemapfile)
    cat = catalogue(catfile, load_all=True)

    # if a raw file and a pipeline file are both given, load them all in
    # if only a raw file is given then the simulation is not from the oslo pipeline
    if rawsimfile:
        # map objects setup
        rawmap = maps(rawsimfile)

        # match catalogue wcs
        cat.match_wcs(rawmap, pipemap, params)


    # trim the catalogue to match the pipeline map
    cat.cull_to_map(pipemap, params, maxsep=2*u.deg)

    # sort the catalogue on Lco if available
    try:
        cat.sort('Lco')

        if params.goalnumcutouts:
            random_mass_subset(cat, params)

        cat.del_extras()
    except AttributeError:
        cat.del_extras()

    # if an output filename is passed, dump the wcs-matched catalogue
    if outcatfile:
        cat.dump(outcatfile)

    return pipemap, cat, rawmap

def sim_setup(pipemapfiles, catfiles, params, rawsimfiles=None, outcatfiles=None):
    """
    wrapper to set up multiple simulated COMAP fields at once
    """
    # list of nones if not saving files
    if not outcatfiles:
        outcatfiles = []
        for i in range(len(pipemapfiles)):
            outcatfiles.append(None)

    # just loop through file list and run sim_field_setup on each
    pipemaps = []
    cats = []
    for i in range(len(pipemapfiles)):
        if rawsimfiles:
            pipemap, cat = sim_field_setup(pipemapfiles[i], catfiles[i],
                                           params, rawsimfile=rawsimfiles[i],
                                           outcatfile=outcatfiles[i])
        else:
            pipemap, cat = sim_field_setup(pipemapfiles[i], catfiles[i], params,
                                           outcatfile=outcatfiles[i])

        pipemaps.append(pipemap)
        cats.append(cat)

    return pipemaps, cats


"""SIGNAL-INJECTION WRAPPERS"""
# **** HAVEN'T BEEN UPDATED TO NEW OBJECTS YET
def sim_inject_field(datfile, simfile, catfile, outfile=None, scale=1.):
    """
    wrapper function -- loads in an actual COMAP map and a simulated halo luminosity LIM
    cube, matches the wcs between the two, and injects the simulated signal into the
    actual map, scaling by 'scale'. Will then save the new map and the associated halo
    catalogue (also wcs-matched) to file.
    Works on a SINGLE COMAP field
    """

    # load the actual data in (this acts as noise)
    datmap = load_map(datfile)

    # load the simulation in (preserving its wcs for now)
    simmap = load_raw_sim(simfile)
    simcat = load_raw_catalogue(catfile, pixel_values=simmap)

    # match wcs
    simmap, datmap, simcat = simcoords_to_mapcoords(simmap, datmap, simcat)

    # inject the signal into the map, beating the noise down by the scale factor
    injected_map = datmap.map / scale + simmap.rawmap
    injected_rms = datmap.rms / scale

    # new map object to store all this info in
    injmap = datmap.copy()
    injmap.sim = injected_map
    injmap.rms = injected_rms
    injmap.simonly = simmap.rawmap
    injmap.datonly = datmap.map

    # save the injected simulation like it's a normal COMAP .h5 product
    ## make a file name if none is given for both the map and the catalogue
    if not outfile:
        datname = datfile.split('/')[-1][:-3]
        simname = simfile.split('seed_')[-1][:-3]
        outfile = 'simcube_' + datname + '_sim_' + simname

        if scale != 1.:
            outfile += '_scale{:.1f}.h5'.format(scale)
        else:
            outfile += '.h5'

    ## save the map
    dump_map(injmap, outfile)

    return injmap, simcat

def sim_inject(datfiles, simfiles, catfiles, outputdir=None, scale=1., trim=None):
    """
    takes input for three different COMAP fields with associated simulations and
    does the injection, matching wcs in the simulation and catalogue to the actual
    data wcs
    --------
    scale:  DIVIDES the input data map (i.e. the noise) by this factor to simulate
            beating the noise down
    trim:   number of catalogue objects to keep (default is all of them). can be
            either a list of integers (one per field) or a single integer (same
            number kept in each field)
    """

    # output path management stuff - save all files to a passed directory
    if not outputdir:
        outputdir = './output/'
    else:
        if outputdir[-1] != '/': outputdir += '/'

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    outfiles = []
    simnames = []
    fieldnames = []
    for i in range(3):
        datname = datfiles[i].split('/')[-1][:-3]
        fieldnames.append('co'+datname.split('co')[-1][0])
        simname = simfiles[i].split('seed_')[-1][:-4]
        simnames.append(simname[:5])
        outfile = outputdir + datname + '_sim_' + simname
        if scale != 1.:
            outfile += '_scale{:.1f}.h5'.format(scale)
        else:
            outfile += '.h5'
        outfiles.append(outfile)

    # come up with a file name for the combined catalogue
    allfieldcatfile = outputdir+'combined_cat_fields_'+'-'.join(fieldnames)+'_seeds_'+'-'.join(simnames)
    if trim:
        if isinstance(trim, (list, tuple, np.ndarray)):
            allfieldcatfile += '_trim_'+'-'.join([str(ti) for ti in trim])
        else:
            allfieldcatfile += '_trim_'+trim
    allfieldcatfile +=  '.npz'

    # individual injections for each field
    fieldcats = []
    for i in range(3):
        fieldmap, fieldcat = sim_inject_field(datfiles[i], simfiles[i], catfiles[i],
                                              outfile=outfiles[i], scale=scale)

        fieldcats.append(fieldcat)

    # combine all the output catalogues together into a single file for ease
    allfieldcat = empty_table()
    if isinstance(trim, (list, tuple, np.ndarray)):
        allfieldcat.z = np.concatenate([cat.z[:trim[i]] for i,cat in enumerate(fieldcats)])
        allfieldcat.ra = np.concatenate([cat.ra[:trim[i]] for i,cat in enumerate(fieldcats)])
        allfieldcat.dec = np.concatenate([cat.dec[:trim[i]] for i,cat in enumerate(fieldcats)])

    elif trim:
        allfieldcat.z = np.concatenate([cat.z[:trim] for cat in fieldcats])
        allfieldcat.ra = np.concatenate([cat.ra[:trim] for cat in fieldcats])
        allfieldcat.dec = np.concatenate([cat.dec[:trim] for cat in fieldcats])

    else:
        allfieldcat.z = np.concatenate([cat.z for cat in fieldcats])
        allfieldcat.ra = np.concatenate([cat.ra for cat in fieldcats])
        allfieldcat.dec = np.concatenate([cat.dec for cat in fieldcats])

    dump_cat(allfieldcat, allfieldcatfile)

    outfiles.append(allfieldcatfile)

    return outfiles

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

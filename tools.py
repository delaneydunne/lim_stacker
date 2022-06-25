from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
import os
import h5py
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", message="invalid value encountered in power")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")


""" OBJECTS AND DICTS AND RELATED CONVENIENCE FUNCTIONS """
class empty_table():
    """
    simple Class creating an empty table
    used for halo catalogue and map instances
    """
    def __init__(self):
        pass

    def copy(self):
        """@brief Creates a copy of the table."""
        return copy.copy(self)

def printdict(dict):
    """
    print a python dict to terminal, testing each variable to see if it has units
    """
    print('{')
    for key in dict.keys():
        if isinstance(dict[key], u.Quantity):
            val = dict[key].value
        else:
            val = dict[key]
        print("'", key, "':", val,",")
    print('}')

def unzip(tablist):
    """
    unzipper to take a list of identical empty_table objects and return arrays containing the contents of each
    individual attribute over the list
    """

    # turn all the individual objects into dicts (to keep only the attributes and
    # their values) if they're not already
    if ~isinstance(tablist[0], dict):
        dictlist = []
        for obj in tablist:
            dictlist.append(vars(obj))

    # dict to be returned
    d = {}
    for k in dictlist[0].keys():
        if isinstance(dictlist[0][k], np.ndarray):
            d[k] = np.stack(list(d[k] for d in dictlist))
        else:
            d[k] = np.array(tuple(list(d[k] for d in dictlist)))

    return d

""" MATH """
def weightmean(vals, rmss, axis=None):
    """
    average of vals, weighted by rmss, over the passed axes
    """
    meanval = np.nansum(vals/rmss**2, axis=axis) / np.nansum(1/rmss**2, axis=axis)
    meanrms = np.sqrt(1/np.nansum(1/rmss**2, axis=axis))
    return meanval, meanrms

def rootmeansquare(vals):
    """
    rms variation in an array
    """
    N = len(vals)
    square = vals**2
    return np.sqrt(np.nansum(square) / N)

def gauss(x, a, b, c):
    """
    1-dimensional Gaussian probability distribution with scaleable amplitude
    """
    return a*np.exp(-(x-b)**2/2/c**2)

""" DOPPLER CONVERSIONS """
def freq_to_z(nuem, nuobs):
    """
    returns a redshift given an observed and emitted frequency
    """
    zval = (nuem - nuobs) / nuobs
    return zval

def nuem_to_nuobs(nuem, z):
    """
    returns the frequency at which an emitted line at a given redshift would be
    observed
    """
    nuobs = nuem / (1 + z)
    return nuobs

def nuobs_to_nuem(nuobs, z):
    """
    returns the frequency at which an observed line at a given redshift would have
    been emitted
    """
    nuem = nuobs * (1 + z)
    return nuem

# def coord_to_pix(coords, comap):
#     """
#     given a coordinate value in degrees, return the (x,y) coordinates
#     according to the map stored in comap
#     """
#     xval = (coords[0] - comap.ra[0]) / (comap.ra[1] - comap.ra[0])
#     yval = (coords[1] - comap.dec[0]) / (comap.dec[1] - comap.dec[0])
#     return (xval, yval)

""" SETUP FUNCTIONS """
def load_map(file, reshape=True):
    """
    loads in a file in the COMAP format, storing everything as arrays in the map class.
    COMAP data are stored with coordinates as the CENTER of each pixel
    """
    # *** give maps their own special class at some point?

    comap = empty_table() # creates empty class to put map info into

    with h5py.File(file, 'r') as mapfile:
        maptemparr = np.array(mapfile.get('map_coadd'))
        rmstemparr = np.array(mapfile.get('rms_coadd'))
        comap.freq = np.array(mapfile.get('freq'))
        comap.ra = np.array(mapfile.get('x'))
        comap.dec = np.array(mapfile.get('y'))

        patch_cent = np.array(mapfile.get('patch_center'))
        comap.fieldcent = SkyCoord(patch_cent[0]*u.deg, patch_cent[1]*u.deg)

        # mark pixels with zero rms and mask them in the rms/map arrays (how the pipeline stores infs)
    comap.badpix = np.where(rmstemparr < 1e-10)
    maptemparr[comap.badpix] = np.nan
    rmstemparr[comap.badpix] = np.nan

    comap.map = maptemparr
    comap.rms = rmstemparr

    if reshape:
        # also reshape into 3 dimensions instead of separating sidebands
        comap.freq = np.reshape(comap.freq, 4*64)
        comap.map = np.reshape(comap.map, (4*64, len(comap.ra), len(comap.dec)))
        comap.rms = np.reshape(comap.rms, (4*64, len(comap.ra), len(comap.dec)))

    # 1-pixel width for each of the axes
    comap.fstep = comap.freq[1] - comap.freq[0]
    comap.xstep = comap.ra[1] - comap.ra[0]
    comap.ystep = comap.dec[1] - comap.dec[0]

    # housekeeping for the arrays - give each axis an index array as well
    comap.x = np.arange(len(comap.ra))
    comap.y = np.arange(len(comap.dec))

    # rearrange so that the stored coordinate coordinate arrays correspond to the
    # bottom right (etc.) of the voxel (currently they're the center)
    comap.freq = comap.freq - comap.fstep / 2
    comap.ra = comap.ra - comap.xstep / 2
    comap.dec = comap.dec - comap.ystep / 2

    # bin edges for each axis for convenience
    comap.freqbe = np.append(comap.freq, comap.freq[-1] + comap.fstep)
    comap.rabe = np.append(comap.ra, comap.ra[-1] + comap.xstep)
    comap.decbe = np.append(comap.dec, comap.dec[-1] + comap.ystep)


    # limits on each axis for easy testing
    comap.flims = (np.min(comap.freq), np.max(comap.freq))
    comap.xlims = (np.min(comap.ra), np.max(comap.ra))
    comap.ylims = (np.min(comap.dec), np.max(comap.dec))

    # *** any other per-field info we need

    return comap

def setup(mapfiles, cataloguefile, params):
    maplist = []
    for mapfile in mapfiles:
        mapinst = load_map(mapfile)

        # calculate the appropriate redshift limits from the freq axis
        zlims = freq_to_z(params.centfreq, np.array(mapinst.flims))
        mapinst.zlims = np.sort(zlims)

        maplist.append(mapinst)

    catdict = {}
    with np.load(cataloguefile) as catfile:
        catdict['z'] = catfile['z']
        catdict['ra'] = catfile['ra']
        catdict['dec'] = catfile['dec']

    catlist = []
    for i in range(len(mapfiles)):
        catinst = field_cull_galaxy_cat(catdict, maplist[i])
        catlist.append(catinst)

    return maplist, catlist

def field_cull_galaxy_cat(galdict, comap, maxsep=3*u.deg):
    """
    takes the full version of the catalogue to be stacked on and cuts to all objects within some
    radius of the given field center
    """
    # *** get rid of skycoord dependence
    # allow you to carry around other arbitrary parameters? ****
    fieldcent = comap.fieldcent
    zlims = np.array(comap.zlims)

    # pull only objects in the field
    fieldcoords = SkyCoord(galdict['ra']*u.deg, galdict['dec']*u.deg)
    fieldsep = fieldcoords.separation(fieldcent)
    fieldidx = np.where(fieldsep < maxsep)[0]

    fieldz_cut = galdict['z'][fieldidx]
    fieldidx = fieldidx[np.where(np.logical_and(fieldz_cut > zlims[0], fieldz_cut < zlims[1]))[0]]

    # save to cat object
    galcat = empty_table()
    galcat.coords = fieldcoords[fieldidx]
    galcat.z = galdict['z'][fieldidx]
    galcat.idx = fieldidx

    # number objects in cat
    galcat.nobj = len(fieldidx)

    return galcat

def make_output_pathnames(params, append=True):
    """
    Uses the input parameters to automatically make a directory to save data
    with an informational name. If there's already a path name passed, uses that one
    """

    sinfo = '_x'+str(params.xwidth)+'f'+str(params.freqwidth)

    if params.savepath and append:
        outputdir = params.savepath + sinfo
    elif not params.savepath:
        outputdir = './stack' + sinfo
    else:
        outputdir = params.savepath

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    params.plotsavepath = outputdir + '/plots'
    params.datasavepath = outputdir + '/data'

    if params.saveplots:
        # make the directories to store the plots and data
        os.makedirs(params.plotsavepath, exist_ok=True)
    if params.savedata:
        os.makedirs(params.datasavepath, exist_ok=True)

    return


""" EXTRA PLOTTING FUNCTIONS """
def plot_mom0(comap, ext=0.95, lognorm=True):

    """
    unsure about the transpose thing
    """

    fig,ax = plt.subplots(1)

    moment0 = weightmean(comap.map, comap.rms, axis=(0))[0] * 1e6
    vext = (np.nanmin(moment0)*ext, np.nanmax(moment0)*ext)
    if lognorm:
        c = ax.pcolormesh(comap.ra, comap.dec, moment0.T,
                          norm=SymLogNorm(linthresh=1, linscale=0.5,
                                          vmin=vext[0], vmax=vext[1]),
                          cmap='PiYG')
    else:
        c = ax.pcolormesh(comap.ra, comap.dec, moment0.T, cmap='PiYG_r')
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')

    cbar = fig.colorbar(c)
    cbar.ax.set_ylabel(r'$T_b \ (\mu K)$')

    return 0

def plot_chan(comap, channel, ext=0.95, lognorm=True):
    fig,ax = plt.subplots(1)
    plotmap = comap.map[channel,:,:] * 1e6
    vext = (np.nanmin(plotmap)*ext, np.nanmax(plotmap)*ext)
    if lognorm:
        c = ax.pcolormesh(comap.ra, comap.dec, plotmap,
                          norm=SymLogNorm(linthresh=1, linscale=0.5,
                                          vmin=vext[0], vmax=vext[1]),
                          cmap='PiYG')
    else:
        c = ax.pcolormesh(comap.ra, comap.dec, plotmap, cmap='PiYG_r')
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')

    cbar = fig.colorbar(c, ax=ax, extend='both')
    cbar.ax.set_ylabel(r'$T_b \ (\mu K)$')

    return 0

""" SETUP FOR SIMS/BOOTSTRAPS """
def field_zbin_stack_output(galidxs, comap, galcat, params):

    usedzvals = galcat.z[galidxs]

    nperbin, binedges = np.histogram(usedzvals, bins=params.nzbins)

    return nperbin, binedges

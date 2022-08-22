from __future__ import print_function
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
from astropy.convolution import Gaussian2DKernel
import os
import sys
import h5py
import csv
import warnings
import copy
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
        return copy.deepcopy(self)

class parameters():
    """
    class creating a custom object used to hold the various stacking parameters
    will autofill defaults
    """

    def __init__(self, paramfile=None):
        """
        assign each of the parameters a default value
        if paramfile is passed, assign each of the parameters their value from paramfile
        paramfile can have as many or as few of the inputs as need be
        """

        modpath = getattr(sys.modules['stacker.tools'], '__file__')
        abspath = modpath.split('tools.py')[0]

        # load defaults in from the separate file that should be in the same directory
        # save them in a directory to do stuff with
        default_dir = {}
        with open(abspath+'param_defaults.py') as f:
            for line in f:
                try:
                    (attr, val) = line.split()
                    default_dir[attr] = val
                except:
                    continue

        # if any of the parameters are not default, replace them here
        if paramfile:
            with open(paramfile) as f:
                for line in f:
                    # if the line in the file contains an actual parameter
                    try:
                        (attr, val) = line.split()
                        # replace the default parameter value with the new one
                        default_dir[attr] = val
                    except:
                        continue

        self.dir = default_dir

        # integer-valued parameters
        for attr in ['xwidth', 'ywidth', 'freqwidth']:
            try:
                val = int(default_dir[attr])
                setattr(self, attr, val)
            except:
                warnings.warn("Parameter '"+attr+"' should be an integer", RuntimeWarning)
                setattr(self, attr, None)

        # float-valued parameters
        for attr in ['centfreq', 'beamwidth']:
            try:
                val = float(default_dir[attr])
                setattr(self, attr, val)
            except:
                warnings.warn("Parameter '"+attr+"' should be a float", RuntimeWarning)
                setattr(self, attr, None)

        # boolean parameters
        for attr in ['cubelet', 'obsunits', 'verbose', 'savedata', 'saveplots', 'plotspace', 'plotfreq']:
            try:
                val = bool(default_dir[attr])
                setattr(self, attr, val)
            except:
                warnings.warn("Parameter '"+attr+"' should be boolean", RuntimeWarning)
                setattr(self, attr, None)

        # optional parameters
        if self.savedata:
            setattr(self, 'savepath', default_dir['savepath'])
            self.make_output_pathnames()
        else:
            setattr(self, 'savepath', None)

        if self.plotspace:
            try:
                setattr(self, 'spacestackwidth', int(default_dir['spacestackwidth']))
            except:
                warnings.warn("Parameter 'spacestackwidth' should be an integer", RuntimeWarning)
        else:
            setattr(self, 'savepath', None)

        if self.plotfreq:
            try:
                setattr(self, 'freqstackwidth', int(default_dir['freqstackwidth']))
            except:
                warnings.warn("Parameter 'freqstackwidth' should be an integer", RuntimeWarning)
        else:
            setattr(self, 'savepath', None)

        # kernel object for the beam
        self.gauss_kernel = Gaussian2DKernel(self.beamwidth)

    def make_output_pathnames(self, append=True):
        """
        Uses the input parameters to automatically make a directory to save data
        with an informational name. If there's already a path name passed, uses that one
        """

        sinfo = '_x'+str(self.xwidth)+'f'+str(self.freqwidth)

        if self.savepath and append:
            outputdir = self.savepath + sinfo
        elif not self.savepath:
            outputdir = './stack' + sinfo
        else:
            outputdir = self.savepath

        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        self.plotsavepath = outputdir + '/plots'
        self.datasavepath = outputdir + '/data'

        if self.saveplots:
            # make the directories to store the plots and data
            os.makedirs(self.plotsavepath, exist_ok=True)
        if self.savedata:
            os.makedirs(self.datasavepath, exist_ok=True)


class catalogue():
    """
    class creating a custom object used to hold galaxy catalogues
    must pass a .npz file to load in data
    """

    def __init__(self):
        pass

    def load(self, inputfile, load_all=False):
        """
        load in data from a .npz catalogue file
        file must have redshift, coordinates -- all other entries in the file will only
        be loaded if load_all=True
        """
        with np.load(inputfile) as f:

            inputdict = dict(f)

            # redshifts
            try:
                self.z = inputdict.pop('z')
            except:
                try:
                    self.z = inputdict.pop('redshift')
                except:
                    warnings.warn('No redshift in input catalogue', RuntimeWarning)

            # coordinates
            try:
                self.coords = SkyCoord(inputdict.pop('ra')*u.deg, inputdict.pop('dec')*u.deg)
            except:
                warnings.warn('No RA/Dec in input catalogue', RuntimeWarning)

            if load_all:
                if len(inputdict) != 0:
                    for attr in inputdict.keys():
                        setattr(self, attr, inputdict[attr])

            self.nobj = len(self.z)
            #*** TYPE OF CATALOGUE FLAG?

    def copy(self):
        """
        creates a deep copy of the object (ie won't overwrite original)
        """
        return copy.deepcopy(self)


    def subset(self, subidx, in_place=False):
        """
        cuts catalogue down to only the catalogue entries at subidx
        """

        if in_place:
            for i in dir(self):
                if i[0] == '_': continue
                try:
                    vals = getattr(self, i)[subidx]
                    setattr(self, i, vals)
                except TypeError:
                    pass
            self.nobj = len(subidx)

        else:
            subset = self.copy()
            for i in dir(self):
                if i[0] == '_': continue
                try:
                    vals = getattr(self, i)[subidx]
                    setattr(subset, i, vals)
                except TypeError:
                    pass
            subset.nobj = len(subidx)
            return subset

    def sort(self, attr):
        """
        sorts catalogue on its attribute attr
        will order so that the max value is index zero
        """

        # pull and sort the array
        tosort = getattr(self, attr)
        sortidx = np.flip(np.argsort(tosort))

        for i in dir(self):
            if i[0] == '_': continue

            try:
                val = getattr(self, i)[sortidx]
                setattr(self, i, val)
            except IndexError:
                val = getattr(self, i)
                setattr(self, i, val)
            except TypeError:
                pass

    def set_nuobs(self, params):
        """
        find observed frequency of each catalogue object
        """
        self.freq = nuem_to_nuobs(params.centfreq, self.z)

    def set_chan(self, comap, params):
        """
        find the frequency channel that each catalogue object will fall into in comap
        """
        try:
            freq = self.freq
        except AttributeError:
            self.set_nuobs(params)

        pixfreq = []
        for i in range(self.nobj):
            objpixfreq = np.max(np.where(comap.freqbe < self.freq[i])[0])
            pixfreq.append(objpixfreq)

        self.chan = np.array(pixfreq)

    def set_pix(self, comap, params):
        """
        find x and y index of the location of each catalogue object in the map
        will also find freq if not already set
        """
        x = []
        y = []
        for i in range(self.nobj):
            objx = np.max(np.where(comap.rabe < self.ra()[i])[0])
            objy = np.max(np.where(comap.decbe < self.dec()[i])[0])

            x.append(objx)
            y.append(objy)

        self.x = np.array(x)
        self.y = np.array(y)

        try:
            _ = self.chan
        except AttributeError:
            self.set_chan(comap, params)


    def cull_to_chan(self, comap, params, chan, in_place=False):
        """
        return a subset of the original cat containing only objects in the passed channel chan
        """
        # if chan isn't already in the catalogue, put it in
        try:
            catchans = self.chan
        except AttributeError:
            self.set_chan(comap, params)

        # indices of catalogue objects corresponding to chan
        inidx = np.where(self.chan == chan)[0]

        # either return a new catalogue object or cut the original one with only
        # objects in chan
        if in_place:
            self.subset(inidx, in_place=True)

        else:
            return self.subset(inidx)


    def cull_to_map(self, comap, params, maxsep = 2*u.deg, in_place=False):
        """
        return a subset of the original cat containing only objects that fall into
        comap
        """

        # objects which fall into the field spatially
        fieldsep = self.coords.separation(comap.fieldcent)
        fieldxbool = fieldsep < maxsep

        # objects which fall into the field spectrally
        try:
            fieldzbool = np.logical_and(self.freq > comap.flims[0], self.freq < comap.flims[1])
        except AttributeError:
            self.set_nuobs(params)
            fieldzbool = np.logical_and(self.freq > comap.flims[0], self.freq < comap.flims[1])

        fieldidx = np.where(np.logical_and(fieldxbool, fieldzbool))[0]

        # either return a new catalogue object or cut the original one with only objects
        # in the field
        if in_place:
            self.subset(fieldidx, in_place=True)
            self.idx = fieldidx
        else:
            mapcat = self.subset(fieldidx)
            mapcat.idx = fieldidx

            return mapcat

    """ RA/DEC CONVENIENCE FUNCTIONS """
    def ra(self):
        return self.coords.ra.deg

    def dec(self):
        return self.coords.dec.deg

    """ COORDINATE MATCHING FUNCTIONS (SIMULATIONS) """
    def match_wcs(self, inmap, outmap, params, in_place=True):
        """
        for simulations -- will adjust the catalogue wcs from matching one map to matching another
        only adjusts ra/dec -- frequency axes should be identical between the two already unless
        inmap is more finely sampled than outmap
        """

        # if the catalogue hasn't already been mapped to inmap, do so
        try:
            _ = self.x
        except AttributeError:
            self.set_pix(inmap, params)

        # check the frequency axis
        if len(inmap.freq) > len(outmap.freq):
            # if the difference isn't integer we've got a problem
            if len(inmap.freq) % len(outmap.freq) != 0:
                warnings.warn('mismatch in number of channels between input and output map',
                              RuntimeWarning)
                return

            subchan_factor = len(inmap.freq) // len(outmap.freq)
            # frequency entries in the catalogue should be fine -- it's just chan
            # that needs to change
            # floor to the nearest integer channel number
            self.chan = self.chan // subchan_factor

        # change the catalogue ra/dec from matching inmap to matching outmap
        ra = self.ra() + inmap.radiff/2 - inmap.ra[0] + outmap.ra[0]
        dec = self.dec() + inmap.decdiff/2 - inmap.dec[0] + outmap.dec[0]

        # map ra and dec
        if in_place:
            self.coords = SkyCoord(ra*u.deg, dec*u.deg)
            return

        else:
            outcat = self.copy()
            outcat.coords = SkyCoord(ra*u.deg, dec*u.deg)
            return outcat

    def del_extras(self):
        for attr in ['Lco', 'M', 'nhalo', 'nu', 'vx', 'vy', 'vz', 'x_pos',
                     'y_pos', 'z_pos', 'zformation']:
            try:
                delattr(self, attr)
            except AttributeError:
                continue

    """ TODOS """
    def print(self):
        """
        ***
        """
        pass

    def dump(self):
        """
        ***
        """
        pass


class maps():
    """
    class containing a custom object used to hold a 3-D LIM map and its associated metadata
    """

    def __init__(self):
        pass

    def load(self, inputfile, reshape=True):
        # this is the COMAP pipeline format currently -- would have to change this if
        # using some other file format
        self.type = 'data'

        # load in from file
        with h5py.File(inputfile, 'r') as file:
            maptemparr = np.array(file.get('map_coadd'))
            rmstemparr = np.array(file.get('rms_coadd'))
            self.freq = np.array(file.get('freq'))
            self.ra = np.array(file.get('x'))
            self.dec = np.array(file.get('y'))

            patch_cent = np.array(file.get('patch_center'))
            self.fieldcent = SkyCoord(patch_cent[0]*u.deg, patch_cent[1]*u.deg)

        # mark pixels with zero rms and mask them in the rms/map arrays (how the pipeline stores infs)
        self.badpix = np.where(rmstemparr < 1e-10)
        maptemparr[self.badpix] = np.nan
        rmstemparr[self.badpix] = np.nan

        self.map = maptemparr
        self.rms = rmstemparr

        if reshape:
            # also reshape into 3 dimensions instead of separating sidebands
            self.freq = np.reshape(self.freq, 4*64)
            self.map = np.reshape(self.map, (4*64, len(self.ra), len(self.dec)))
            self.rms = np.reshape(self.rms, (4*64, len(self.ra), len(self.dec)))

        # build the other convenience coordinate arrays, make sure the coordinates map to
        # the correct part of the voxel
        self.setup_coordinates()


    def load_sim(self, inputfile):
        """
        loads in a limlam_mocker simulation in raw format instead of a pipeline simulation
        will not adjust coordinates or anything, but will format the output object to be
        identical to the regular data objects
        """
        self.type = 'raw simulation'

        # load in from file
        with np.load(inputfile, allow_pickle=True) as simfile:
            # sims output uK, data in K. stack functions all deal w K so convert
            self.map = simfile['map_cube'] / 1e6
            # these are bin CENTERS also
            self.freq = simfile['map_frequencies']
            self.ra = simfile['map_pixel_ra']
            self.dec = simfile['map_pixel_dec']

        # flip frequency axis and rearrange so freq axis is the first in the map
        self.map = np.swapaxes(self.map, 0, -1)
        self.map = np.swapaxes(self.map, 1, 2)
        self.freq = np.flip(self.freq)
        self.map = np.flip(self.map, axis=0)

        # build the other convenience coordinate arrays, make sure the coordinates map to
        # the correct part of the voxel
        self.setup_coordinates()


    def setup_coordinates(self):
        """
        takes an input map (in the correct orientation) and adds the binedge, bin center, etc
        coordinate arrays
        """

        # 1-pixel width for each of the axes
        self.fstep = self.freq[1] - self.freq[0]
        self.xstep = self.ra[1] - self.ra[0]
        self.ystep = self.dec[1] - self.dec[0]

        # housekeeping for the arrays - give each axis an index array as well
        self.x = np.arange(len(self.ra))
        self.y = np.arange(len(self.dec))

        # rearrange so that the stored coordinate coordinate arrays correspond to the
        # bottom right (etc.) of the voxel (currently they're the center)
        self.freq = self.freq - self.fstep / 2
        self.ra = self.ra - self.xstep / 2
        self.dec = self.dec - self.ystep / 2

        # bin edges for each axis for convenience
        self.freqbe = np.append(self.freq, self.freq[-1] + self.fstep)
        self.rabe = np.append(self.ra, self.ra[-1] + self.xstep)
        self.decbe = np.append(self.dec, self.dec[-1] + self.ystep)

        # limits on each axis for easy testing
        self.flims = (np.min(self.freq), np.max(self.freq))
        self.xlims = (np.min(self.ra), np.max(self.ra))
        self.ylims = (np.min(self.dec), np.max(self.dec))



    """ COORDINATE MATCHING FUNCTIONS (FOR SIMULATIONS) """
    def rebin_freq(self, goalmap, in_place=True):
        """
        simulation pipeline takes a map that's more finely sampled in the frequency axis than
        the science resolution. This will rebin input maps to match the output ones
        """

        # first test to make sure the desired rebinning makes sense
        if len(self.freq) < len(goalmap.freq):
            warnings.warn('Input frequency axis less finely-sampled than goal map', RuntimeWarning)

        elif len(self.freq) % len(goalmap.freq) != 0:
            warnings.warn('Input number of channels is not an integer multiple of goal', RuntimeWarning)

        chan_factor = len(self.freq) // len(goalmap.freq)

        # if the map is real (i.e. there exists an RMS map), weightmean to combine the map
        try:
            inmap = self.map.reshape((chan_factor, -1, len(self.ra), len(self.dec)))
            inrms = self.rms.reshape((chan_factor, -1, len(self.ra), len(self.dec)))
            rebinmap, rebinrms = weightmean(inmap, inrms, axis=0)

        except AttributeError:
            # otherwise just a regular mean
            inmap = self.map.reshape((chan_factor, -1, len(self.ra), len(self.dec)))
            rebinmap = np.nanmean(inmap, axis=0)
            rebinrms = None

        # tack these new arrays on
        if in_place:
            self.map = rebinmap
            if rebinrms:
                self.rms = rebinrms
        else:
            outmap = self.copy()
            outmap.map = rebinmap
            if rebinrms:
                self.rms = rebinrms

        # also do the frequency axis
        rebinfreq = self.freq[::chan_factor]
        rebinfreqbe = self.freqbe[::chan_factor]
        rebinfstep = rebinfreq[1] - rebinfreq[0]

        if in_place:
            self.freq = rebinfreq
            self.freqbe = rebinfreqbe
            self.fstep = rebinfstep

        else:
            outmap.freq = rebinfreq
            outmap.freqbe = rebinfreqbe
            outmap.fstep = rebinfstep

        if not in_place:
            return outmap


    def match_wcs(self, goalmap, params, in_place=True):
        """
        for simulations -- will adjust the map wcs from whatever is already in self to wcs matching
        that of goalmap. frequency axis should already be identical unless self is more finely sampled
        than goalmap. if in_place=False will return a copy
        """

        # first make sure the sizes are the same for each map
        if len(self.ra) != len(goalmap.ra) or len(self.dec) != len(goalmap.dec):
            warnings.warn('Input/output WCS axes have different dimensions', RuntimeWarning)
            pass

        # if in_place is true, just replace the relevant axes
        if in_place:
            self.ra = goalmap.ra
            self.rabe = goalmap.rabe
            self.dec = goalmap.dec
            self.decbe = goalmap.dec

        else:
            # otherwise, make a copy and return it
            outmap = self.copy()
            outmap.ra = goalmap.ra
            outmap.rabe = goalmap.rabe
            outmap.dec = goalmap.dec
            outmap.decbe = goalmap.decbe

        # rebin the freq axis if necessary
        if len(self.freq) != len(goalmap.freq):
            if in_place:
                self.rebin_freq(goalmap, in_place=True)
            else:
                outmap.rebin_freq(goalmap, in_place=True)

        # return outmap if needbe
        if not in_place:
            return outmap


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
        print("'{}': {:.3e},".format(key, val))
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
    else:
        dictlist = tablist

    # dict to be returned
    d = {}
    for k in dictlist[0].keys():
        if isinstance(dictlist[0][k], np.ndarray):
            d[k] = np.stack(list(d[k] for d in dictlist))
        else:
            d[k] = np.array(tuple(list(d[k] for d in dictlist)))

    return d

def dict_saver(indict, outfile, strip_units=True):
    """
    function to save a dictionary to an output .csv file
    can't handle dicts with more than one value per key, but can handle dicts
    with units -- it will strip the units off if strip_units is true so they're
    easier to open on the other end

    can also handle a list of dicts with identical keys
    """

    if isinstance(indict, list):
        # if a list of dicts is given, have to iterate through them
        unitless_dict_list = []
        for idict in indict:
            if strip_units:
                # if we want units gone, this needs to be done for each dict individually
                # before printing
                unitless_dict = {}
                for (key, val) in idict.items():
                    if isinstance(val, u.Quantity):
                        key = key + ' (' + val.unit.to_string() + ')'
                        unitless_dict[key] = val.value
                    else:
                        unitless_dict[key] = val
            else:
                # otherwise just copy them over
                unitless_dict = idict.copy()
            # new list of cleaned dicts to print to file
            unitless_dict_list.append(unitless_dict)

    # if there's only one dict given, don't need to iterate -- just clean it if
    # necessary
    else:
        if strip_units:
            unitless_dict = {}
            for (key, val) in indict.items():
                if isinstance(val, u.Quantity):
                    key = key + ' (' + val.unit.to_string() + ')'
                    unitless_dict[key] = val.value
                else:
                    unitless_dict[key] = val
        else:
            unitless_dict = indict.copy()

    # will have a floating single dict with keys to reference no matter what, so
    # print those first
    with open(outfile, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(unitless_dict.keys()))
        writer.writeheader()
        # then, either print many rows or only one depending on what was passed
        if isinstance(indict, list):
            for idict in unitless_dict_list:
                writer.writerow(idict)
            # return the cleaned dict list
            return unitless_dict_list

        else:
            writer.writerow(unitless_dict)
            # return the cleaned dict
            return unitless_dict


""" MATH """
def weightmean(vals, rmss, axis=None):
    """
    average of vals, weighted by rmss, over the passed axes
    """
    meanval = np.nansum(vals/rmss**2, axis=axis) / np.nansum(1/rmss**2, axis=axis)
    meanrms = np.sqrt(1/np.nansum(1/rmss**2, axis=axis))
    return meanval, meanrms

def globalweightmean(vals, rmss, axis=None):
    """
    average of vals, weighted by rmss, over the passed axes
    difference from weightmean is that this function will calculate a global rms
    for each object to be combined
    """
    if axis == None:
        globrms = np.nanmean(rmss)
    else:
        naxis = np.arange(len(rmss.shape))
        naxis = tuple(np.delete(naxis, axis))
        globrms = np.nanmean(rmss, axis=naxis)

    meanval = (np.nansum(vals, axis=axis) / globrms**2) / (globrms**2)
    meanrms = np.sqrt(1/(1/globrms**2))

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
    """
    wrapper function to load in data and set up objects for a stack run
    """
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

def field_setup(mapfile, catfile, params):
    """
    wrapper function to set up for a single-field stack run
    """
    # load in the map
    mapinst = load_map(mapfile)

    # load in the catalogue
    catinst = catalogue()
    catinst.load(catfile)

    # clip the catalogue to the field
    catinst.cull_to_map(mapinst, params, maxsep=2*u.deg, in_place=True)

    return mapinst, catinst

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


""" OBJECT HANDLING FUNCTIONS """
def catobj_in_chan(channel, cat, comap):
    """
    creates a new catalogue object containing only entries in the correct map channel
    doesn't affect the og catalogue object
    """

    # if chan is already in trcat, just slice based on that
    # otherwise have to map frequencies to channels and then slice
    try:
        inidx = np.where(cat.chan == channel)[0]

    except:
        obsfreq = nuem_to_nuobs(115.27, cat.z)

        if trawmap.freqbe[0] < trawmap.freqbe[1]:
            freqmin, freqmax = comap.freqbe[channel], comap.freqbe[channel+1]
        else:
            freqmin, freqmax = comap.freqbe[channel+1], comap.freqbe[channel]

        inidx = np.where(np.logical_and(obsfreq < freqmax,
                                        obsfreq >= freqmin))[0]

    # slice out a new catalogue object only containing the entries in the correct
    # map channel
    chancat = empty_table()
    try:
        chancat.ra = cat.coords.ra.deg[inidx]
    except:
        chancat.ra = cat.ra[inidx]

    try:
        chancat.dec = cat.coords.dec.deg[inidx]
    except:
        chancat.dec = cat.dec[inidx]

    # copy over all array parameters from the catalogue (from dongwoo chung)
    # generic bc there are a couple of different ways the catalogue could be set up
    # *** move this to a specific catalogue object
    for i in dir(cat):
        if i[0] == '_': continue

        try:
            setattr(chancat, i, getattr(cat, i)[inidx])
        except TypeError:
            pass

    chancat.nobj = len(inidx)

    return chancat

def sort_cat(cat, attr):
    """
    sorts catalogue on its attribute attr
    will order so that the max value is index zero
    this is done in-place
    """

    # pull and sort the array
    tosort = getattr(cat, attr)
    sortidx = np.flip(np.argsort(tosort))

    # temporary object to hold unsorted values
    tempcat = cat.copy()

    for i in dir(cat):
        if i[0] == '_': continue

        try:
            setattr(cat, i, getattr(tempcat, i)[sortidx])
        except TypeError:
            pass

    del(tempcat)
    return cat


""" SETUP FOR SIMS/BOOTSTRAPS """
def field_zbin_stack_output(galidxs, comap, galcat, params):

    usedzvals = galcat.z[galidxs]

    nperbin, binedges = np.histogram(usedzvals, bins=params.nzbins)

    return nperbin, binedges

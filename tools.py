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

    def print(self):
        attrlist = []
        for i in dir(self):
            if i[0]=='_': continue
            elif i == 'copy': continue
            else: attrlist.append(i)
        print(attrlist)

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
        for attr in ['cubelet', 'obsunits', 'verbose', 'savedata', 'saveplots', 'plotspace', 'plotfreq', 'plotcubelet']:
            try:
                val = bool(default_dir[attr])
                setattr(self, attr, val)
            except:
                warnings.warn("Parameter '"+attr+"' should be boolean", RuntimeWarning)
                setattr(self, attr, None)

        # make sure you're not trying to plot a cubelet if you're not actually making one
        if not self.cubelet:
            self.plotcubelet = False
            warnings.warn("plotcubelet==True when cubelet==False -- set plotcubelet to False", RuntimeWarning)

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

        # if the default one was accidentally saved, get rid of it
        if self.savepath != 'stack_output'+sinfo and os.path.exists('stack_output'+sinfo):
                os.rmdir('stack_output'+sinfo+'/data')
                os.rmdir('stack_output'+sinfo+'/plots')
                os.rmdir('stack_output'+sinfo)
        elif self.savepath != 'stack_output' and os.path.exists('stack_output'):
            os.rmdir('stack_output/data')
            os.rmdir('stack_output/plots')
            os.rmdir('stack_output')

        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        self.plotsavepath = outputdir + '/plots'
        self.datasavepath = outputdir + '/data'

        if self.saveplots:
            # make the directories to store the plots and data
            os.makedirs(self.plotsavepath, exist_ok=True)
        if self.savedata:
            os.makedirs(self.datasavepath, exist_ok=True)

    def info(self):
        """
        quick printer to give a summary of the settings of a params object
        """
        print("Parameters object for an intensity-mapping stack run")
        print("-------------")
        print("Stack parameters")
        print("-------------")
        print("\t (xwidth, ywidth, freqwidth): ({},{},{})".format(self.xwidth,
                                                                  self.ywidth,
                                                                  self.freqwidth))
        print("-------------")
        print("Map parameters")
        print("-------------")
        print("\t Central frequency: {}".format(self.centfreq))
        print("\t Beam STD (pix): {}".format(self.beamwidth))
        print("-------------")
        print("Stacking metaparameters")
        print("-------------")
        print("\t Cubelet stacking: {}".format(self.cubelet))
        print("\t Returning observer units: {}".format(self.obsunits))
        print("\t Verbose output: {}".format(self.verbose))
        print("\t Saving stack data: {}".format(self.savedata))
        if self.savedata:
            print("\t\t to path: "+self.datasavepath)
        print("\t Saving plots: {}".format(self.saveplots))
        if self.saveplots:
            print("\t\t to path: "+self.plotsavepath)
        print("\t Radius of output spatial image: {} pix".format(self.spacestackwidth))
        print("\t Diameter of output spectrum: {} channels".format(self.freqstackwidth))
        print("-------------")


class catalogue():
    """
    class creating a custom object used to hold galaxy catalogues
    must pass a .npz file to load in data
    """

    def __init__(self, inputfile=None, load_all=False):
        if inputfile:
            self.load(inputfile, load_all=load_all)
        else:
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


    def subset(self, subidx, in_place=True):
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


    def cull_to_chan(self, comap, params, chan, in_place=True):
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
            return self.subset(inidx, in_place=False)


    def cull_to_map(self, comap, params, maxsep = 2*u.deg):
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
        self.subset(fieldidx, in_place=True)
        self.idx = fieldidx

    """ RA/DEC CONVENIENCE FUNCTIONS """
    def ra(self):
        return self.coords.ra.deg

    def dec(self):
        return self.coords.dec.deg

    """ COORDINATE MATCHING FUNCTIONS (SIMULATIONS) """
    def match_wcs(self, inmap, outmap, params):
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
        ra = self.ra() - inmap.ra[0] + outmap.ra[0]
        dec = self.dec() - inmap.dec[0] + outmap.dec[0]

        # map ra and dec
        self.coords = SkyCoord(ra*u.deg, dec*u.deg)


    def del_extras(self):
        for attr in ['Lco', 'M', 'nhalo', 'nu', 'vx', 'vy', 'vz', 'x_pos',
                     'y_pos', 'z_pos', 'zformation']:
            try:
                delattr(self, attr)
            except AttributeError:
                continue

    def info(self):
        """
        prints a quick summary of what's going on in the catalogue (so you don't
        have to print(dir()) every time)
        """
        # directory containing all the attributes in the catalogue
        catdir = dir(self)

        print("Catalogue object for an intensity-mapping stack run")
        print("Contains {} total objects".format(self.nobj))
        print("-------------")
        if 'Lco' in catdir:
            _ = self.Lco
            print("Halo luminosities in range {:.3e} to {:.3e}".format(np.min(self.Lco),
                                                                       np.max(self.Lco)))
        else:
            print("No halo mass/luminosity information")
        print("-------------")
        print("RA in range {:.3f} to {:.3f} deg".format(np.min(self.ra()), np.max(self.ra())))
        print("Dec in range {:.3f} to {:.3f} deg".format(np.min(self.dec()), np.max(self.dec())))
        print("Redshift in range {:.3e} to {:.3e}".format(np.min(self.z), np.max(self.z)))
        print("-------------")
        print("Catalogue also includes:")
        for i in catdir:
            if i[0] == '_': continue
            elif i == 'coords': continue
            elif i == 'nobj': continue
            elif i == 'z': continue

            elif isinstance(getattr(self, i), np.ndarray):
                print(i + ": ({}, {})".format(np.min(getattr(self, i)), np.max(getattr(self, i))))
        print("-------------")


    def dump(self, outfile):
        """
        save the catalogue info in a file readable by catalogue again
        """
        np.savez(outfile, z=self.z, ra=self.ra(), dec=self.dec())



class maps():
    """
    class containing a custom object used to hold a 3-D LIM map and its associated metadata
    """

    def __init__(self, inputfile=None, reshape=True):
        if inputfile:
            if inputfile[-2:] == 'h5':
                self.load(inputfile, reshape=reshape)
            elif inputfile[-3:] == 'npz':
                self.load_sim(inputfile)
            else:
                warnings.warn('Unrecognized input file type', RuntimeWarning)
        else:
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
    def rebin_freq(self, goalmap):
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
        self.map = rebinmap
        if rebinrms:
            self.rms = rebinrms

        # also do the frequency axis
        self.freq = self.freq[::chan_factor]
        self.freqbe = self.freqbe[::chan_factor]
        self.fstep = rebinfreq[1] - rebinfreq[0]


    def match_wcs(self, goalmap, params):
        """
        for simulations -- will adjust the map wcs from whatever is already in self to wcs matching
        that of goalmap. frequency axis should already be identical unless self is more finely sampled
        than goalmap. will always do this in place
        """

        # first make sure the sizes are the same for each map
        if len(self.ra) != len(goalmap.ra) or len(self.dec) != len(goalmap.dec):
            warnings.warn('Input/output WCS axes have different dimensions', RuntimeWarning)
            pass

        #  just replace the relevant axes
        self.ra = goalmap.ra
        self.rabe = goalmap.rabe
        self.dec = goalmap.dec
        self.decbe = goalmap.dec

        # also give self the field center given in goalmap
        self.fieldcent = goalmap.fieldcent

        # rebin the freq axis if necessary
        if len(self.freq) != len(goalmap.freq):
            self.rebin_freq(goalmap)


    def dump(self, outfile):
        """
        save the map object to an h5 file formatted like the COMAP pipeline output
        """

        # undo the coordinate shift so it doesn't happen twice when it's reloaded
        outfreq = self.freq + self.fstep / 2
        outra = self.ra + self.xstep / 2
        outdec = self.dec + self.ystep / 2

        # save to hdf5 file (slightly more barebones than the actual output)
        with h5py.File(outfile, 'w') as f:
            dset = f.create_dataset('map_coadd', data = self.sim, dtype='float64')
            dset = f.create_dataset('rms_coadd', data = self.rms, dtype='float64')
            dset = f.create_dataset('freq', data = outfreq, dtype='float64')
            dset = f.create_dataset('x', data = outra, dtype='float64')
            dset = f.create_dataset('y', data = outdec, dtype='float64')

            patchcent = (self.fieldcent.ra.deg, self.fieldcent.dec.deg)
            dset = f.create_dataset('patch_center', data = patchcent, dtype='float64')

    def info(self):
        """
        prints out information about the map object
        """
        if self.type == 'data':
            print("Map object read in from an actual pipeline run")
        elif self.type == 'raw simulation':
            print(" Map object from a raw limlam_mocker simulation")
        else:
            print("Map object from unknown source")
        print("-------------")
        print("Field Center:")
        print("({:.3f}, {:.3f}) deg".format(self.fieldcent.ra.deg, self.fieldcent.dec.deg))
        print("-------------")
        print("WCS:")
        print("map shape is (freq, Dec, RA) ({}, {}, {})".format(*self.map.shape))
        print("RA extent: ({:.3f}, {:.3f}) deg".format(np.max(self.rabe), np.min(self.rabe)))
        print("\t step size: {:.3f} deg".format(self.xstep))
        print("Dec extent: ({:.3f}, {:.3f}) deg".format(np.max(self.decbe), np.min(self.decbe)))
        print("\t step size: {:.3f} deg".format(self.ystep))
        print("Frequency extent: ({:.3f}, {:.3f}) GHz".format(np.max(self.freqbe), np.min(self.freqbe)))
        print("\t channel width: {:.3f} GHz".format(self.fstep))
        print("-------------")
        print("Map extent:")
        print("Max T: {:.3e} uK".format(np.nanmax(self.map)*1e6))
        print("Min T: {:.3e} uK".format(np.nanmin(self.map)*1e6))
        print("-------------")


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
def field_setup(mapfile, catfile, params):
    """
    wrapper function to set up for a single-field stack run
    """
    # load in the map
    mapinst = maps(mapfile)

    # load in the catalogue
    catinst = catalogue(catfile)

    # clip the catalogue to the field
    catinst.cull_to_map(mapinst, params, maxsep=2*u.deg)

    return mapinst, catinst

def setup(mapfiles, cataloguefile, params):
    """
    wrapper function to load in data and set up objects for a stack run
    """
    maplist = []
    catlist = []
    for i in range(len(mapfiles)):
        mapinst, catinst = field_setup(mapfiles[i], cataloguefile, params)
        maplist.append(mapinst)
        catlist.append(catinst)

    return maplist, catlist





""" SETUP FOR SIMS/BOOTSTRAPS """
def field_zbin_stack_output(galidxs, comap, galcat, params):

    usedzvals = galcat.z[galidxs]

    nperbin, binedges = np.histogram(usedzvals, bins=params.nzbins)

    return nperbin, binedges

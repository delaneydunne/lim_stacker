from __future__ import print_function
# from .plottools import *
# from .stack import *
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
from astropy.convolution import Gaussian2DKernel, convolve
from astropy import wcs
from astropy.cosmology import FlatLambdaCDM
from pixell import utils
from reproject import reproject_adaptive
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

        modpath = getattr(sys.modules['lim_stacker.tools'], '__file__')
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
        for attr in ['xwidth', 'ywidth', 'freqwidth', 'usefeed', 'voxelhitlimit']:
            try:
                val = int(default_dir[attr])
                setattr(self, attr, val)
            except:
                warnings.warn("Parameter '"+attr+"' should be an integer", RuntimeWarning)
                setattr(self, attr, None)
        # condition for pulling a specific feed
        if self.usefeed > 20:
            self.usefeed = False

        # float-valued parameters
        for attr in ['centfreq', 'beamwidth', 'fitmeanlimit', 'voxelrmslimit']:
            try:
                val = float(default_dir[attr])
                setattr(self, attr, val)
            except:
                warnings.warn("Parameter '"+attr+"' should be a float", RuntimeWarning)
                setattr(self, attr, None)

        # boolean parameters
        for attr in ['cubelet', 'obsunits', 'rotate', 'lowmodefilter', 'chanmeanfilter',
                     'specmeanfilter', 'verbose', 'returncutlist', 'savedata', 'saveplots',
                     'savefields', 'plotspace', 'plotfreq', 'plotcubelet', 'physicalspace',
                     'adaptivephotometry']:
            try:
                val = default_dir[attr] == 'True'
                setattr(self, attr, val)
            except:
                warnings.warn("Parameter '"+attr+"' should be boolean", RuntimeWarning)
                setattr(self, attr, None)

        # make sure you're not trying to plot a cubelet if you're not actually making one
        if not self.cubelet:
            self.plotcubelet = False
            warnings.warn("plotcubelet==True when cubelet==False -- set plotcubelet to False", RuntimeWarning)

        # optional parameters
        if self.rotate:
            try:
                setattr(self, 'rotseed', int(default_dir['rotseed']))
            except:
                self.rotseed = 12345
                warnings.warn("Missing random seed for rotation. Using 12345 as default", RuntimeWarning)
            self.rng = np.random.default_rng(self.rotseed)


        try:
            setattr(self, 'fitnbeams', int(default_dir['fitnbeams']))
        except:
            self.fitnbeams = 3
            warnings.warn("Missing number of beams for cutout fitting. Using 3 as default", RuntimeWarning)

        try:
            setattr(self, 'fitmasknbeams', int(default_dir['fitmasknbeams']))
        except:
            self.fitmasknbeams = 1
            warnings.warn("Missing number of beams for cutout fitting aperture mask. Using 1 as default", RuntimeWarning)
        try:
            setattr(self, 'freqmaskwidth', int(default_dir['freqmaskwidth']))
        except:
            self.freqmaskwidth = 1
            warnings.warn("Missing number of apertures to mask for calculating spectral mean. Using 1 as default", RuntimeWarning)
        try:
            setattr(self, 'frequsewidth', int(default_dir['frequsewidth']))
        except:
            self.frequsewidth = 10
            warnings.warn("Missing number of apertures for calculating spectral mean. Using 10 as default", RuntimeWarning)


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

        # units for plotting
        try:
            setattr(self, 'plotunits', default_dir['plotunits'])
        except:
            warnings.warn("Parameter 'plotunits' should be a string. defaulting to linelum units", RuntimeWarning)
            setattr(self, 'plotunits', 'linelum')

        # cosmology to use
        if default_dir['cosmo'] == 'comap':
            setattr(self, 'cosmo', FlatLambdaCDM(H0=70*u.km / (u.Mpc*u.s), Om0=0.286, Ob0=0.047))
        else:
            warnings.warn("Don't recognize parameter 'cosmo'. defaulting to COMAP values")
            setattr(self, 'cosmo', FlatLambdaCDM(H0=70*u.km / (u.Mpc*u.s), Om0=0.286, Ob0=0.047))


        # number of cutouts can be either a list or an int
        if default_dir['goalnumcutouts'][0] == '[':
            ncuts = default_dir['goalnumcutouts'][1:-1].split(',')
            self.goalnumcutouts = [int(val) for val in ncuts]
        else:
            try:
                setattr(self, 'goalnumcutouts', int(default_dir['goalnumcutouts']))
            except:
                self.goalnumcutouts = None

        # kernel object for the beam
        self.gauss_kernel = Gaussian2DKernel(self.beamwidth / (2*np.sqrt(2*np.log(2))))

    def make_output_pathnames(self, append=True):
        """
        Uses the input parameters to automatically make a directory to save data
        with an informational name. If there's already a path name passed, uses that one
        """

        # add extra info to the filename because i will forget it
        sinfo = '_x'+str(self.xwidth)+'f'+str(self.freqwidth)
        if self.rotate:
            sinfo += '_rot'
        if self.lowmodefilter:
            sinfo += '_lmfilt'
        if self.chanmeanfilter:
            sinfo += '_cmfilt'
        if self.lowmodefilter or self.chanmeanfilter:
            sinfo += '_r'+str(self.fitnbeams)+'m'+str(self.fitmasknbeams)

        if self.savepath and append:
            outputdir = self.savepath + sinfo
        elif not self.savepath:
            outputdir = './stack' + sinfo
        else:
            outputdir = self.savepath

        # if the default one was accidentally saved, get rid of it
        if self.savepath != 'stack_output'+sinfo and os.path.exists('stack_output'+sinfo):
            try:
                os.rmdir('stack_output'+sinfo+'/data')
                os.rmdir('stack_output'+sinfo+'/plots')
                os.rmdir('stack_output'+sinfo)
            except:
                pass

        elif self.savepath != 'stack_output' and os.path.exists('stack_output'):
            try:
                os.rmdir('stack_output/data')
                os.rmdir('stack_output/plots')
                os.rmdir('stack_output')
            except:
                pass

        # make the new output directory
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        self.savepath = outputdir
        self.plotsavepath = outputdir + '/plots'
        self.datasavepath = outputdir + '/data'
        self.cubesavepath = outputdir + '/plots/cubelet'

        # if saving fields individually, set up for that
        if self.savefields:
            fields = ['/field1', '/field2', '/field3']

        # if bootstrapping, adjust those file names too ********
        try:
            self.itersavefile = outputdir + '/' + self.itersavefile
            self.nitersavefile = outputdir + '/' + self.nitersavefile
        except:
            pass

        if self.saveplots:
            # make the directories to store the plots and data
            if self.savefields:
                # individual ones for each field if doing that
                for field in fields:
                    os.makedirs(self.plotsavepath+field, exist_ok=True)
                    if self.plotcubelet:
                        os.makedirs(self.cubesavepath+field, exist_ok=True)
            else:
                os.makedirs(self.plotsavepath, exist_ok=True)
                if self.plotcubelet:
                    os.makedirs(self.cubesavepath, exist_ok=True)
        if self.savedata:
            if self.savefields:
                for field in fields:
                    os.makedirs(self.datasavepath+field, exist_ok=True)
            else:
                os.makedirs(self.datasavepath, exist_ok=True)
            

    def copy(self):
        """
        returns a deep copy of the params object (ie won't link back to the original)
        """
        return copy.deepcopy(self)

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
        print("\t\t extra cubelet plots: {}".format(self.plotcubelet))
        print("\t Radius of output spatial image: {} pix".format(self.spacestackwidth))
        print("\t Diameter of output spectrum: {} channels".format(self.freqstackwidth))
        print("-------------")
        print("Filtering parameters")
        print("-------------")
        print("\t Global mean subtraction from spectrum: {}".format(self.specmeanfilter))
        if self.specmeanfilter:
            print("\t\t Numbers of apertures: ({},{})".format(self.freqmaskwidth, self.frequsewidth))
        print("\t Per-channel mean subtraction: {}".format(self.chanmeanfilter))
        if self.chanmeanfilter:
            print("\t\t Numbers of beams: ({}, {})".format(self.fitmasknbeams, self.fitnbeams))
        print("\t Low-order polynomial removal: {}".format(self.lowmodefilter))
        if self.lowmodefilter:
            print("\t\t Numbers of beams: ({}, {})".format(self.fitmasknbeams, self.fitnbeams))
        print("-------------")

    def create_sensmap_bootstrap(self, sensfilepath):
        """
        reads in the hetdex sensitivity map and sets it up for bootstrapping (so it only has to
        be done once)
        **note these are pretty tight cuts -- if you're relaxing the rms/hit constraints might 
        need to relax these as well
        """

        # field 1
        with np.load(sensfilepath+'sensitivity_field_1_processed.npz') as f:
            hexsens = f['sens']
            ras = f['ra']
            decs = f['dec']
            
        self.field_1_sensmap = (ras, decs, hexsens)

        # field 2
        with np.load(sensfilepath+'sensitivity_field_2_processed.npz') as f:
            hexsens = f['sens']
            ras = f['ra']
            decs = f['dec']
            
        self.field_2_sensmap = (ras, decs, hexsens)

        # field 3
        with np.load(sensfilepath+'sensitivity_field_3_processed.npz') as f:
            hexsens = f['sens']
            ras = f['ra']
            decs = f['dec']
            
        self.field_3_sensmap = (ras, decs, hexsens)

        # redshift
        with np.load(sensfilepath+'sensitivity_redshift_average.npz') as f:
            zbins = f['bins']
            zprobs = f['prob']
        self.redshift_sensmap = (zbins, zprobs)




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
                # fix skycoords wrapping thing around zero by just adding 3 deg to all ra coordinates if this is a simulation
                ra = inputdict.pop('ra')
                dec = inputdict.pop('dec')

                if np.any(ra < 0):
                    ra = ra + 3

                self.coords = SkyCoord(ra*u.deg, dec*u.deg)
            except:
                warnings.warn('No RA/Dec in input catalogue', RuntimeWarning)

            if load_all:
                if len(inputdict) != 0:
                    for attr in inputdict.keys():
                        setattr(self, attr, inputdict[attr])

            self.nobj = len(self.z)
            # index in the actual catalogue file passed
            self.catfileidx = np.arange(self.nobj)
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
                except (TypeError, IndexError):
                    pass
            self.nobj = len(subidx)

        else:
            subset = self.copy()
            for i in dir(self):
                if i[0] == '_': continue
                try:
                    vals = getattr(self, i)[subidx]
                    setattr(subset, i, vals)
                except (TypeError, IndexError):
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
            # account for SkyCoord doing 2pi rotations on its own now
            ra = self.ra()[i]
            if ra > 250:
                ra -= 360
            objx = np.max(np.where(comap.rabe < ra)[0])
            objy = np.max(np.where(comap.decbe < self.dec()[i])[0])

            x.append(objx)
            y.append(objy)

        self.x = np.array(x)
        self.y = np.array(y)

        try:
            _ = self.chan
        except AttributeError:
            self.set_chan(comap, params)

    def z_offset(self, mean, scatter, params, type='z', in_place=True, verbose=True):
        """
        randomly offset the velocities in the catalogue using a gaussian kernal
        if type is z, mean/scatter are redshifts
        if type is freq, mean/scatter are (observed) frequencies
        if type is vel, mean/scatter are velocities
        if not in_place, returns a copy
        if verbose, will also spit out the passed offset and mean as freq/velocity/redshift if not given
        """

        rng = params.rng

        # if verbose:
        #     zmean = np.mean(catinst.z)
        #     if type == 'vel':
        #         rvmean = const.c*((1+zmean)**2 - 1) / ((1+zmean)**2 + 1)
        #         nrvmean = rvmean + mean 

                

        if in_place:
            # just scatter the original
            if type == 'z':
                offset_redshifts(self, mean, scatter, rng)
            elif type == 'vel':
                offset_velocities(self, mean, scatter, rng)
            elif type == 'freq':
                offset_frequencies(self, mean, scatter, rng)
            else:
                print('????. Not offsetting')
            return
        else:
            # scatter a copy so the original is kept
            catinst = self.copy()
            if type == 'z':
                offset_redshifts(catinst, mean, scatter, rng)
            elif type == 'vel':
                offset_velocities(catinst, mean, scatter, rng)
            elif type == 'freq':
                offset_frequencies(catinst, mean, scatter, rng)
            else:
                print('????. Not offsetting')
            return catinst
        

    def add_false_positives(self, percent, comap, params, in_place=True):
        """
        replace x percent of the catalogue with (randomly generated) false postivies hits
        -------
        INPUTS:
        -------
            percent:   the percentage of the catalogut that should be false positves
            comap:     the map object that will be stacked on (as a percentage, ie 100 for all)
            params:    the parameters object
            in_place:  (bool; default=True) if False, will make a copy to insert FPs into
                        otherwise, just changes self
        """

        # calculate the actual number of false positives necessary to get percent
        num_fp = int(self.nobj * percent / 100)
        num_keep = int(self.nobj * (100-percent)/100)

        # generate that number of random coordinates
        ravals = params.rng.uniform(comap.xlims[0], comap.xlims[1], size=num_fp)
        decvals = params.rng.uniform(comap.ylims[0], comap.ylims[1], size=num_fp)
        coordvals = SkyCoord(ravals*u.deg, decvals*u.deg)

        zlims = freq_to_z(params.centfreq, np.array([comap.flims[0], comap.flims[1]]))
        zvals = params.rng.uniform(zlims[1], zlims[0], size=num_fp)

        # generate random indices to replace with false positives
        fpidx = params.rng.integers(0, self.nobj, num_fp)

        if in_place:
            # store the indices that are false positives in the catalogue object for future reference
            self.fpidx = fpidx 
            # change the coords at the FP indices to the random ones
            self.coords[fpidx] = coordvals 
            self.z[fpidx] = zvals 

            return 
        
        else:
            # generate a copy of self to store the fp values in
            newcat = self.copy()
            # store the fp indices for future reference 
            newcat.fpidx = fpidx 
            # change the coords at the FP indices to the random ones
            newcat.coords[fpidx] = coordvals 
            newcat.z[fpidx] = zvals 

            return newcat



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

        if len(fieldidx) == 0:
            print('Warning: cull removed all objects from field')

        # either return a new catalogue object or cut the original one with only objects
        # in the field
        self.subset(fieldidx, in_place=True)
        self.idx = fieldidx

    def observation_cull(self, params, lcat_cutoff, goal_nobj, rngseed=None):
        """
        cut a simulated catalog based on its observational parameters: cut to objects only above a certain luminosity
        (sensitivity limit of the catalog) and then randomly select N objects from that cut list
        uses:
            lcat_cutoff: the lower limit on catalog luminosity to include (in Lsun)
            goal_nobj: number of catalog objects to include once the cut is made
        """

        # cut by luminosity
        goodidx = np.where(self.Lcat > lcat_cutoff)[0]
        self.subset(goodidx)

        # select nobj random objects from the leftover catalog, shuffling their indices randomly
        if goal_nobj > 0:
            if not rngseed:
                rngseed = params.rotseed
            rng = np.random.default_rng(rngseed)
            keepidx = rng.choice(self.nobj, goal_nobj, replace=False)
            self.subset(keepidx)
            
        if params.verbose: print('\n\t%d halos remain after observability cuts' % self.nhalo)


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
        ** new version using interpolation **
        """

        # if the catalogue hasn't already been mapped to inmap, do so
        try:
            _ = self.x
        except AttributeError:
            self.set_pix(inmap, params)

        self.cull_to_map(inmap, params) 

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

        # set up rotation matrix to zero
        incentra, incentdec = inmap.fieldcent.ra.deg, inmap.fieldcent.dec.deg
        inra = utils.rotmatrix(np.deg2rad(-incentra), raxis='z')
        indec = utils.rotmatrix(np.deg2rad(-incentdec), raxis='y')
        inrotmatrix = inra @ indec

        # set up rotation matrix from zero to new coordinate center
        outcentra, outcentdec = outmap.fieldcent.ra.deg, outmap.fieldcent.dec.deg
        outra = utils.rotmatrix(np.deg2rad(outcentra), raxis='z')
        outdec = utils.rotmatrix(np.deg2rad(outcentdec), raxis='y')
        outrotmatrix = outra @ outdec

        # put coordinates to rotate into pixell format
        invector = utils.ang2rect((np.deg2rad(self.ra()), np.deg2rad(self.dec())))

        # send input coordinates to equator
        midvector = utils.rect2ang(inrotmatrix @ invector)
        midra, middec = midvector 

        # flip ra and dec once at the equator
        midvector = utils.ang2rect((middec, -midra))

        # send equator coordinates to output field
        outvector = utils.rect2ang(outrotmatrix @ midvector)

        # save to catalog object
        outra, outdec = outvector 
        outra, outdec = np.rad2deg(outra), np.rad2deg(outdec)
        outra = outra + outmap.ystep
        self.coords = SkyCoord(outra*u.deg, -outdec*u.deg)



    def match_wcs_old(self, inmap, outmap, params):
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

    def __init__(self, params, inputfile=None, reshape=True):
        if inputfile:
            if inputfile[-2:] == 'h5':
                self.load(inputfile, params, reshape=reshape)
            elif inputfile[-3:] == 'npz':
                self.load_sim(inputfile, params)
            else:
                warnings.warn('Unrecognized input file type', RuntimeWarning)
        else:
            pass

    def copy(self):
        return copy.deepcopy(self)

    def load(self, inputfile, params, reshape=True):

        """
        map shape is [4, 64(256), 120, 120]: sidebands, freq, dec, ra
        """

        # this is the COMAP pipeline format currently -- would have to change this if
        # using some other file format
        self.type = 'data'

        # load in from file
        with h5py.File(inputfile, 'r') as file:

            # coordinate info (not feed-dependent)
            self.freq = np.array(file.get('freq'))
            self.ra = np.array(file.get('x'))
            self.dec = np.array(file.get('y'))

            if isinstance(params.usefeed, bool):
                maptemparr = np.array(file.get('map_coadd'))
                rmstemparr = np.array(file.get('rms_coadd'))
                hittemparr = np.array(file.get('nhit_coadd'))

                # account for new naming conventions
                # **** new maps can't handle single feeds yet bc i don't think those
                # conventions exist as
                if not np.any(maptemparr):
                    maptemparr = np.array(file.get('map'))
                    rmstemparr = np.array(file.get('rms'))
                    hittemparr = np.array(file.get('nhit'))
                    self.freq = np.array(file.get('freq_centers'))
                    self.ra = np.array(file.get('ra_centers'))
                    self.dec = np.array(file.get('dec_centers'))

                # even newer naming conventions
                if not np.any(rmstemparr):
                    rmstemparr = np.array(file.get('sigma_wn_coadd'))
                    self.freq = np.array(file.get('freq_centers'))
                    self.ra = np.array(file.get('ra_centers'))
                    self.dec = np.array(file.get('dec_centers'))

            else:
                feedidx = params.usefeed - 1
                if params.verbose:
                    print('loading feed {} only'.format(params.usefeed))
                # load each of the individual feed maps
                maptemparr = np.array(file.get('map'))[feedidx,:,:,:,:]
                rmstemparr = np.array(file.get('rms'))[feedidx,:,:,:,:]
                hittemparr = np.array(file.get('nhit'))[feedidx,:,:,:,:]

                if not np.any(self.freq):
                    self.freq = np.array(file.get('freq_centers'))
                    self.ra = np.array(file.get('ra_centers'))
                    self.dec = np.array(file.get('dec_centers'))

                if not np.any(rmstemparr):
                    rmstemparr = np.array(file.get('sigma_wn'))[feedidx,:,:,:,:]
                    print(rmstemparr.shape)

                # if going per-feed, need to knock the hit limit way down
                params.voxelhitlimit /= 19
                # flag the feed you're using
                self.feed = params.usefeed

            patch_cent = np.array(file.get('patch_center'))
            self.fieldcent = SkyCoord(patch_cent[0]*u.deg, patch_cent[1]*u.deg)

        # mark pixels with zero rms and mask them in the rms/map arrays (how the pipeline stores infs)
        mapbadpix = np.logical_or(rmstemparr < 1e-13, rmstemparr > params.voxelrmslimit)
        mapbadpix = np.logical_or(mapbadpix, ~np.isfinite(rmstemparr))
        # also mark anything with less than 10 000 hits (another way to clean off map edges)
        hitbadpix = np.logical_or(hittemparr < params.voxelhitlimit, ~np.isfinite(hittemparr))
        self.badpix = np.where(np.logical_or(mapbadpix, hitbadpix))
        maptemparr[self.badpix] = np.nan
        rmstemparr[self.badpix] = np.nan
        hittemparr[self.badpix] = 0

        self.map = maptemparr
        self.rms = rmstemparr
        self.hit = hittemparr
        self.unit = 'K'

        if reshape:
            chanpersb = self.map.shape[1]
            # also reshape into 3 dimensions instead of separating sidebands
            self.freq = np.reshape(self.freq, 4*chanpersb)
            self.map = np.reshape(self.map, (4*chanpersb, len(self.ra), len(self.dec)))
            self.rms = np.reshape(self.rms, (4*chanpersb, len(self.ra), len(self.dec)))
            self.hit = np.reshape(self.hit, (4*chanpersb, len(self.ra), len(self.dec)))

        # build the other convenience coordinate arrays, make sure the coordinates map to
        # the correct part of the voxel
        self.setup_coordinates()

        # newest iteration flips the ra axis, so undo that:
        if self.xstep < 0:
            self.xstep = -self.xstep
            self.ra = np.flip(self.ra)
            self.rabe = np.flip(self.rabe)
            self.map = np.flip(self.map, axis=-1)
            self.rms = np.flip(self.rms, axis=-1)
            self.hit = np.flip(self.hit, axis=-1)

        # move some things to params to keep the info handy
        params.nchans = self.map.shape[0]
        params.chanwidth = np.abs(self.freq[1] - self.freq[0])

    """UNIT CONVERSIONS"""
    def to_flux(self):
        """ converts from temperature units to flux units. won't do anything if the unit
            isn't already in K"""
        
        if self.unit != 'K':
            print('need units to be K, and current units are '+self.unit)
            return
        
        # correct for primary beam response
        self.map /= 0.72
        self.rms /= 0.72

        # actual COMAP beam
        beam_fwhm = 4.5*u.arcmin
        sigma_x = beam_fwhm / (2 * np.sqrt(2 * np.log(2)))
        sigma_y = sigma_x
        omega_B = (2 * np.pi * sigma_x * sigma_y).to(u.sr)

        # voxel solid angle
        # l_vox = 2*u.arcmin
        # omega_B = (l_vox**2).to(u.sr)

        # central frequency of each individual spectral channel
        self.freqbc = self.fstep / 2 + self.freq
        freqvals = np.tile(self.freqbc, (self.map.shape[2], self.map.shape[1], 1)).T * u.GHz

        # calculate fluxes in Jy
        Svals = rayleigh_jeans(self.map*u.K, freqvals, omega_B)
        Srmss = rayleigh_jeans(self.rms*u.K, freqvals, omega_B)

        # multiply by the channel width in km/s
        delnus = (self.fstep* u.GHz / freqvals * const.c).to(u.km/u.s)

        Snu_Delnu = Svals * delnus
        dSnu_Delnu = Srmss * delnus

        self.map = Snu_Delnu.value
        self.rms = dSnu_Delnu.value
        self.unit = 'flux'

        return
    
    def to_linelum(self, params):

        if self.unit == 'K':
            self.to_flux()

        elif self.unit != 'flux':
            print('need flux or temperature units')
            return 
        
        # put into the appropriate astropy units
        self.map = self.map * u.Jy * u.km/u.s
        self.rms = self.rms * u.Jy * u.km/u.s

        nuobs = np.tile(self.freqbc, (self.map.shape[2], self.map.shape[1], 1)).T * u.GHz

        # find redshift from nuobs:
        zval = freq_to_z(params.centfreq*u.GHz, nuobs) #*****************

        # luminosity distance in Mpc
        DLs = params.cosmo.luminosity_distance(zval)

        # line luminosity
        linelum = const.c**2 / (2*const.k_B) * self.map * DLs**2 / (nuobs**2 * (1+zval)**3)
        dlinelum = const.c**2 / (2*const.k_B) * self.rms * DLs**2 / (nuobs**2 * (1+zval)**3)

        # fix units
        linelum = linelum.to(u.K*u.km/u.s*u.pc**2)
        dlinelum = dlinelum.to(u.K*u.km/u.s*u.pc**2)

        # store in object
        self.map = linelum.value
        self.rms = dlinelum.value
        self.unit = 'linelum'

        return



    def load_sim(self, inputfile, params):
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
            try:
                # if an RMS value is saved, use that
                self.rms = np.ones(self.map.T.shape) * simfile['sigma'] / 1e6
            except KeyError:
                # if not, dummy RMS array
                self.rms = np.ones(self.map.T.shape)

        # if the simulaton is centered around zero, add 2deg to the ra axis so it doesn't 
        # wrap weird in skycoord
        if np.any(self.ra < 0):
            self.ra = self.ra + 3

        # flip frequency axis and rearrange so freq axis is the first in the map
        self.map = np.swapaxes(self.map, 0, -1)
        # self.map = np.swapaxes(self.map, 1, 2) #*******
        self.freq = np.flip(self.freq)
        self.map = np.flip(self.map, axis=0)

        # build the other convenience coordinate arrays, make sure the coordinates map to
        # the correct part of the voxel
        self.setup_coordinates()

        # move some things to params to keep the info handy
        params.nchans = self.map.shape[0]
        params.chanwidth = np.abs(self.freq[1] - self.freq[0])

        # field center 
        self.fieldcent = SkyCoord(self.ra[len(self.ra) // 2]*u.deg, self.dec[len(self.ra) // 2]*u.deg)

        # units
        self.unit = 'K'


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
    def rebin_freq(self, goalmap, params):
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
            inmap = self.map.reshape((chan_factor, -1, len(self.ra), len(self.dec)), order='F')
            inrms = self.rms.reshape((chan_factor, -1, len(self.ra), len(self.dec)), order='F')
            rebinmap, rebinrms = weightmean(inmap, inrms, axis=0)

        except AttributeError:
            # otherwise just a regular mean
            inmap = self.map.reshape((chan_factor, -1, len(self.ra), len(self.dec)), order='F')
            rebinmap = np.nanmean(inmap, axis=0)
            rebinrms = None

        # tack these new arrays on
        self.map = rebinmap
        if rebinrms:
            self.rms = rebinrms

        # also do the frequency axis
        self.freq = self.freq[::chan_factor]
        self.freqbe = self.freqbe[::chan_factor]
        self.fstep = self.freq[1] - self.freq[0]

        # and fix channel width in params
        params.chanwidth = np.abs(self.fstep)

    def rebin_freq_byfactor(self, factor, params, in_place=True):
        """ 
        rebin in frequency by a given factor and not to match to another map
        """

        # reshape to make rebinning easier
        inmap = self.map.reshape((factor, -1, len(self.ra), len(self.dec)), order='F')
        inrms = self.rms.reshape((factor, -1, len(self.ra), len(self.dec)), order='F')
        # rebin by weighted meaning
        rebinmap, rebinrms = weightmean(inmap, inrms, axis=0)

        fstep = (self.freq[1] - self.freq[0])*factor
        # housekeeping channel width in params
        params.chanwidth = np.abs(fstep)

        if in_place:
            self.map = rebinmap
            self.rms = rebinrms 

            # housekeeping frequency axes
            self.freq = self.freq[::factor]
            self.freqbe = self.freqbe[::factor]
            self.fstep = fstep

        else:
            binnedmap = self.copy()

            binnedmap.map = rebinmap
            binnedmap.rms = rebinrms 

            # housekeeping frequency axes
            binnedmap.freq = self.freq[::factor]
            binnedmap.freqbe = self.freqbe[::factor]
            binnedmap.fstep = fstep

            return binnedmap
        
    def rebin_space_byfactor(self, factor, params, in_place=False):
        """
        rebin in the spatial directions by a given factor ********
        """
        pass
        

    def upgrade(self, factor, params, in_place=False):
        """
        oversample the spatial axes by a factor of factor. ***implement this in the spectral axis as well
        currently doesn't interpolate at all, just repeats
        look into the pixell structure where the function is defined seperately***
        """

        # oversample the input map
        bigmap = np.repeat(self.map, factor, axis=1).repeat(factor, axis=2)

        # new metainfo for spatial coordinates
        bigx = np.arange(bigmap.shape[1])
        bigy = np.arange(bigmap.shape[2])

        bigxstep = self.xstep / factor
        bigystep = self.ystep / factor 

        bigra = self.ra[0] + bigx * bigxstep 
        bigdec = self.dec[0] + bigy * bigystep 

        bigrabe = self.rabe[0] + bigx * bigxstep 
        bigdecbe = self.decbe[0] + bigy * bigystep 

        if in_place:
            # populate the object with the new values
            self.map = bigmap 
            self.x = bigx 
            self.y = bigy 
            self.xstep = bigxstep 
            self.ystep = bigystep 
            self.ra = bigra 
            self.dec = bigdec 
            self.rabe = bigrabe 
            self.decbe = bigdecbe

            return 

        else:
            # create a new object populated with the new values 
            outmap = self.copy()
            outmap.map = bigmap
            outmap.x = bigx 
            outmap.y = bigy 
            outmap.xstep = bigxstep 
            outmap.ystep = bigystep
            outmap.ra = bigra 
            outmap.dec = bigdec 
            outmap.rabe = bigrabe 
            outmap.decbe = bigdecbe

            return outmap

    def cosmic_volume_spacing(self, goalres, params, oversamp=5):
        """
        evenly space a maps object in proper cosmic distance
        -------
        inputs:
            self: stacker.maps object
            goalres: 
            oversamp: factor by which the map should be oversampled while reprojecting
        
        """
        
        # add a bin centers attr for the frequency direction
        # and redshift axis
        self.freqbc = self.fstep / 2 + self.freq
        self.z = freq_to_z(115.27, self.freqbc)
        
        # conversion factor for each channel (assuming constant across map)
        chanscales = params.cosmo.kpc_proper_per_arcmin(self.z)
        
        # actual COMAP beam
        beam_fwhm = 4.5*u.arcmin
        sigma_x = beam_fwhm / (2 * np.sqrt(2 * np.log(2)))
        sigma_y = sigma_x
        omega_B = (2 * np.pi * sigma_x * sigma_y)
        
        # beam fwhm in Mpc
        sigmazmpc = (sigma_x * chanscales).to(u.Mpc)
        
        # kernel to put them all on the same size beam
        sigmamax = sigmazmpc[-1]
        sigmaconvmpc = np.sqrt(sigmamax**2 - sigmazmpc**2)
        sigmaconv = (sigmaconvmpc / chanscales).to(u.arcmin)
        
        # oversample
        bigmap = self.upgrade(5, params)
        
        # kernel in pixels
        sigmaconvpix = sigmaconv / ((bigmap.xstep*u.deg).to(u.arcmin))
        
        # width of each channel in Mpc
        chansizes = (chanscales * bigmap.xstep*u.deg).to(u.Mpc)
        
        # set up original wcs: empty wcs thing to store it in
        bigwcsdict = {"CTYPE1": 'RA---CAR', 
                    "CTYPE2": 'DEC--CAR'}
        # populate it with the correct values ***THESE ARE WRONG********
        bigwcs = wcs.WCS(bigwcsdict)
        centidx = np.array([len(bigmap.x)//2, len(bigmap.y)//2])
        bigwcs.wcs.crpix = centidx
        bigwcs.wcs.crval = np.array([bigmap.ra[centidx[0]], bigmap.dec[centidx[1]]])
        bigwcs.wcs.cdelt = np.array([bigmap.xstep, bigmap.ystep])
        
        # goal width
        if not type(goalres) == u.quantity.Quantity:
            goalres *= u.Mpc
            
        # factor by which to resample
        resampfacs = (goalres / chansizes).value
        
        # new degree scale of each channel
        newcdelt = np.array([bigmap.xstep*resampfacs, bigmap.ystep*resampfacs])
        
        # current width of the map
        mapwidth = bigmap.map.shape[1] * bigmap.xstep
        
        # what is this in the new cdelt pixels in each channel
        mapsizes = mapwidth / newcdelt
        mapsize = int(np.round(np.max(mapsizes)))
        mapcent = mapsize // 2
        
        # set up output wcs
        outwcs = copy.deepcopy(bigwcs)
        outwcs.wcs.crpix = np.array([mapcent, mapcent])
        
        
        # loop through channels, reconvolve and reproject
        rprcmaplist = []
        outwcslist = []
        for freq in range(len(bigmap.freq)):
            # reconvolve
            kernel = Gaussian2DKernel(sigmaconvpix[freq].value)
            rcchan = convolve(bigmap.map[freq,:,:], kernel, fill_value=np.nan)
            
            # adjust output wcs
            chanwcs = copy.deepcopy(outwcs)
            chanwcs.wcs.cdelt = np.array([newcdelt[0][freq], newcdelt[1][freq]])
            outwcslist.append(chanwcs)
            
            # reproject
            rprcchan, _ = reproject_adaptive((rcchan, bigwcs), chanwcs,
                                            shape_out=(mapsize,mapsize),
                                            kernel='gaussian', boundary_mode='strict')
            rprcmaplist.append(rprcchan)
            
            if params.verbose:
                if freq % 10 == 0:
                    print(freq)
        rprcmap = np.array(rprcmaplist)
        
        # set up 3d map object with variable pixel scales
        outmap = bigmap.copy()
        outmap.map = rprcmap
        outmap.x = np.arange(mapsize)
        outmap.y = np.arange(mapsize)
        
        outmap.rms = np.ones(np.shape(outmap.map)) # *********
        
        raidx = outmap.x - mapcent
        decidx = outmap.y - mapcent
        newxstep = []
        newystep = []
        newra = []
        newdec = []
        for freq in range(len(outmap.freq)):
            
            cdelts = outwcslist[freq].wcs.cdelt
            chanxstep = cdelts[0]
            chanystep = cdelts[1]
            
            chanra = outwcslist[0].wcs.crval[0] + raidx * chanxstep
            chandec = outwcslist[0].wcs.crval[1] + decidx * chanystep
            
            newxstep.append(np.ones(238)*chanxstep)
            newystep.append(np.ones(238)*chanystep)
            newra.append(chanra)
            newdec.append(chandec)
        
        newxstep = np.array(newxstep)
        newystep = np.array(newystep)
        ra = np.array(newra)
        dec = np.array(newdec)
        
        outmap.xstep = newxstep[:,0]
        outmap.ystep = newystep[:,0]
        outmap.ra = ra
        outmap.dec = dec
        outmap.rabc = ra - newxstep / 2
        outmap.decbc = dec - newystep / 2
        
        # *** save spacing in mpc also
        # *** do this in-place
        
        
        return outmap

        
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
def minmax(vals):
    """
    returns (np.nanmin(vals), np.nanmax(vals)) to get extrema in a single function call
    """

    return np.array([np.nanmin(vals), np.nanmax(vals)])

def weightmean(vals, rmss, axis=None, weights=None):
    """
    average of vals, weighted by rmss, over the passed axes
    default is over a a fully flattened array if no axes are passed
    will be default weight by inverse variance only, but if 'weights' are passed then 
    will also weight by whatever that is
    """
    if np.any(weights):
        weights = weights / rmss**2 # *** probably going to have to worry about the shape of the weights array
    else:
        weights = 1/rmss**2

    meanval = np.nansum(vals*weights, axis=axis) / np.nansum(1*weights, axis=axis)
    meanrms = np.sqrt(1/np.nansum(1*weights, axis=axis))

    # meanrms = (np.nansum(vals**2*weights, axis=axis)/np.nansum(weights, axis=axis) - meanval**2)
    # meanrms *= (np.nansum(weights, axis=axis)**2) / ((np.nansum(weights, axis=axis)**2) - np.nansum(weights**2, axis=axis))

    # meanrms = np.sqrt(meanrms)

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
    a: amplitude
    b: mean
    c: standard deviation
    """
    return a*np.exp(-(x-b)**2/2/c**2)

""" UNIT CONVERSIONS """
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

""" SIMULATION UNIT CONVERSION """
def simlum_to_stacklum(simlum, stackout, params):
    """
    INPUTS:
    simlum: luminosity (in Lsun) from the input catalogue
    stackout: output stacked cube

    RETURNS:
    outlum: the luminosity from the input catalogue converted into the measured luminosity
            (in K km/s) that would correspond to in the stack
    frac: the percentage of this output luminosity that was actually measured in the stack
    """
    # first Lsun to Ico
    convfac = 4.0204e-2 # Jy/sr per Lsol/Mpc/Mpc/GHz
    DLs = params.cosmo.luminosity_distance(stackout.z_mean) # luminosity distances
    Ico     = convfac * simlum/4/np.pi/(DLs.value)**2/(1+stackout.z_mean)**2/params.chanwidth

    # channel widths as velocities
    delnus = (params.chanwidth*u.GHz / (stackout.nuobs_mean*u.GHz)*const.c).to(u.km/u.s)
    
    # flux value
    flux = Ico * delnus * params.freqwidth * u.Jy
    
    # line luminosity
    linelum = const.c**2 / (2*const.k_B) * flux * DLs**2 / ((stackout.nuobs_mean*u.GHz)**2 * (1+stackout.z_mean)**3)
    
    # beam adjustment, units
    outlum = linelum.to(u.K*u.km/u.s*u.pc**2) * 0.72

    frac = stackout.linelum / outlum.value
    
    return outlum, frac

""" SIMULATION OFFSETTING """
def offset_velocities(catinst, meanoff, scatter, rng):
    """
    **** fix this. it is wrong
    randomly offsets the redshift of the catalogue objects with a gaussian kernel
    with mean value meanoff and standard deviation scatter (given as VELOCITIES). 
    RNG is the input random number generator for consistency
    """
    # get recession velocities from the catalog redshifts
    rec_vels = const.c * ((1+catinst.z)**2 - 1) / ((1+catinst.z)**2 + 1)
    rec_vels = rec_vels.to(u.km/u.s)

    # random list of velocity offsets
    off_vels = rng.normal(loc=meanoff, scale=scatter, size=catinst.nobj) * u.km/u.s

    # add these in and generate a new list of redshifts
    new_vels = off_vels + rec_vels 
    new_z = np.sqrt((const.c+new_vels) / (const.c-new_vels)) - 1

    catinst.z = new_z

def offset_redshifts(catinst, meanoff, scatter, rng):
    """
    randomly offsets the redshift of the catalogue objects with a gaussian kernel
    with mean value meanoff and standard deviation scatter (given directly as redshifts). 
    RNG is the input random number generator for consistency
    this is based off (delta z)/(1+z) (like in fig 6 of paper draft)
    """
    # random list of redshift offsets
    off_zs = rng.normal(loc=meanoff, scale=scatter, size=catinst.nobj)

    # add these in and generate a new list of redshifts 
    new_zs = off_zs*(1+catinst.z) + catinst.z

    catinst.z = new_zs

def offset_frequencies(catinst, meanoff, scatter, rng):
    """
    randomly offsets the redshift of the catalogue objects with a gaussian kernel
    with mean value meanoff and standard deviation scatter (given as FREQUENCIES). 
    RNG is the input random number generator for consistency
    """
    
    # observed frequency of each of the objects
    obs_freqs = nuem_to_nuobs(115.27, catinst.z)
    
    # generate a distribution of offset frequencies first
    off_freqs = rng.normal(loc=meanoff, scale=scatter, size=catinst.nobj)
    
    # add these in and generate a new list of redshifts
    new_freqs = off_freqs + obs_freqs
    new_z = freq_to_z(115.27, new_freqs)
    
    catinst.z = new_z


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
def field_setup(mapfile, catfile, params, trim_cat=True, sim_cat=False, lcat_cutoff=None, goal_nobj=None):
    """
    wrapper function to set up for a single-field stack run
    *** tidy this up again -- put simulation parameters into params**
    """
    # load in the map
    mapinst = maps(params, inputfile=mapfile)

    # load in the catalogue
    if not sim_cat:
        catinst = catalogue(catfile)
        # clip the catalogue to the field
        catinst.cull_to_map(mapinst, params, maxsep=2*u.deg)
    else:
        catinst = catalogue(catfile, load_all=True)
        catinst.observation_cull(params, lcat_cutoff, goal_nobj)

    # adjust the beam to match the actual size of the spaxels
    params.beamwidth = params.beamwidth / (mapinst.xstep*u.deg).to(u.arcmin).value
    params.gauss_kernel = Gaussian2DKernel(params.beamwidth / (2*np.sqrt(2*np.log(2))))

    # additional trimming
    if trim_cat:
        print('trimming catalog')
        # trim the catalogs down to match the actual signal in the maps
        goodidx = np.where(~np.isnan(np.nanmean(mapinst.map, axis=0)))
        raminidx, ramaxidx = np.min(goodidx[1]), np.max(goodidx[1])+1
        decminidx, decmaxidx = np.min(goodidx[0]), np.max(goodidx[0])+1

        if ramaxidx >= mapinst.x[-1]:
            ramaxidx = -1
        
        if decmaxidx >= mapinst.y[-1]:
            decmaxidx = -1

        ramin, ramax = mapinst.ra[[raminidx, ramaxidx]]
        decmin, decmax = mapinst.dec[[decminidx, decmaxidx]]

        catidxra = np.logical_and(catinst.ra() > ramin, catinst.ra() < ramax)
        catidxdec = np.logical_and(catinst.dec() > decmin, catinst.dec() < decmax)
        catidx = np.where(np.logical_and(catidxra, catidxdec))[0]
        
        catinst.subset(catidx)

    return mapinst, catinst

def setup(mapfiles, cataloguefile, params, trim_cat=True):
    """
    wrapper function to load in data and set up objects for a stack run
    accepts either a list of per-field catalogue files or one big one
    if there's only one map file, use field_setup
    """
    maplist = []
    catlist = []
    for i in range(len(mapfiles)):
        if isinstance(cataloguefile, (list, tuple, np.ndarray)):
            mapinst, catinst = field_setup(mapfiles[i], cataloguefile[i], params, trim_cat=trim_cat)
        else:
            mapinst, catinst = field_setup(mapfiles[i], cataloguefile, params, trim_cat=trim_cat)
        maplist.append(mapinst)
        catlist.append(catinst)

    # adjust the stored beam model to be in pixels
    params.beamwidth = params.beamwidth / (maplist[0].xstep*u.deg).to(u.arcmin).value
    params.gauss_kernel = Gaussian2DKernel(params.beamwidth / (2*np.sqrt(2*np.log(2))))

    return maplist, catlist


""" CUBE SLICERS """
# convenience functions
def aperture_collapse_cubelet_freq(cvals, crmss, params, recent=0):
    """
    take a 3D cubelet cutout and collapse it along the frequency axis to be an average over the
    stack aperture frequency channels
    either cutout is an empty_table instance or a list of [vals, rmss]
    """

    # indexes of the channels to include
    lcfidx = (cvals.shape[0] - params.freqwidth) // 2 + recent
    cfidx = (lcfidx, lcfidx + params.freqwidth)

    # collapsed image
    # cutim, imrms = weightmean(cvals[cfidx[0]:cfidx[1],:,:],
    #                           crmss[cfidx[0]:cfidx[1],:,:], axis=0)
    cutim = np.nansum(cvals[cfidx[0]:cfidx[1],:,:], axis=0)
    imrms = np.sqrt(np.nansum(crmss[cfidx[0]:cfidx[1],:,:]**2, axis=0))

    return cutim, imrms

def aperture_collapse_cubelet_space(cvals, crmss, params, recentx=0, recenty=0):
    """
    take a 3D cubelet cutout and collapse it along the spatial axis to be an average over the stack
    aperture spaxels (ie make a spectrum)
    """

    # indices of the x and y axes
    lcxidx = (cvals.shape[1] - params.xwidth) // 2 + recentx
    lcyidx = (cvals.shape[2] - params.ywidth) // 2 + recenty
    cxidx = (lcxidx, lcxidx + params.xwidth)
    cyidx = (lcyidx, lcyidx + params.ywidth)

    # clip out values to stack
    fpixval = cvals[:, cyidx[0]:cyidx[1], cxidx[0]:cxidx[1]]
    frmsval = crmss[:, cyidx[0]:cyidx[1], cxidx[0]:cxidx[1]]

    cutspec, specrms = weightmean(fpixval, frmsval, axis=(1,2))

    return cutspec, specrms


def padder(cubelet, rmslet, params):

    xypad = params.xwidth
    fpad = params.freqwidth

    padcubelet = np.pad(cubelet, ((fpad,fpad), (xypad,xypad), (xypad,xypad)), mode='constant', constant_values=np.nan)
    padrmslet = np.pad(rmslet, ((fpad,fpad), (xypad,xypad), (xypad,xypad)), mode='constant', constant_values=np.nan)

    return padcubelet, padrmslet

def cubelet_collapse_pointed(cubelet, rmslet, newcentpix, params, collapse=True):

    if cubelet.shape[0] == params.freqstackwidth*2+1:
        cubelet, rmslet = padder(cubelet, rmslet, params)

    # goal center voxel of the padded cube
    newfcent = newcentpix[0] + params.freqwidth
    newxcent = newcentpix[1] + params.xwidth
    newycent = newcentpix[2] + params.xwidth

    # number of pixels other than the center to include in each axis
    foff = (params.freqwidth - 1) // 2
    xoff = (params.xwidth - 1) // 2
    yoff = xoff

    # indices to keep
    cfidx = (newfcent - foff, newfcent + foff + 1)
    cxidx = (newxcent - xoff, newxcent + xoff + 1)
    cyidx = (newycent - yoff, newycent + yoff + 1)

    apcutout = cubelet[cfidx[0]:cfidx[1], cxidx[0]:cxidx[1], cyidx[0]:cyidx[1]]
    apcutrms = rmslet[cfidx[0]:cfidx[1], cxidx[0]:cxidx[1], cyidx[0]:cyidx[1]]

    if collapse:
        apval, aprms = weightmean(apcutout, apcutrms, axis=(1,2))
        apval = apval * 3 * 3
        aprms = aprms * 3 * 3
        apval = np.nansum(apval)
        aprms = np.sqrt(np.nansum(aprms**2))
    else:
        apval, aprms = apcutout, apcutrms

    return apval, aprms

def aperture_vid(cubelet, rmslet, params):

    pcube, prms = padder(cubelet, rmslet, params)

    outvallist, outdvallist = [], []
    for i in np.arange(1,cubelet.shape[0]-1):
        for j in np.arange(1, cubelet.shape[1]-1):
            for k in np.arange(1, cubelet.shape[2]-1):
                val, dval = cubelet_collapse_pointed(pcube, prms, (i, j, k), params)

                outvallist.append(val)
                outdvallist.append(dval)

    return np.array(outvallist).flatten(), np.array(outdvallist).flatten()


""" SETUP FOR SIMS/BOOTSTRAPS """
def field_zbin_stack_output(galidxs, comap, galcat, params):

    usedzvals = galcat.z[galidxs]

    nperbin, binedges = np.histogram(usedzvals, bins=params.nzbins)

    return nperbin, binedges

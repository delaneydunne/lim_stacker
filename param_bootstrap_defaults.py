""" size of the stack aperture used to calcuate values """
# number of pixels (RA)
xwidth 3
# number of pixels (Dec)
ywidth 3
# number of frequency channels at map freq resolution
freqwidth 1

""" properties of the map """
# CO(1-0) emitted frequency (GHz)
centfreq 115.27
# std of a gaussian 2D beam in pixels
beamwidth 1

""" stacking metaparameters """
 # save cubelets instead of separate spatial/spectral stacks for diagnostic
 # images. this will combine cutouts iteratively instead of returning the
 # full list of them
cubelet False
# return physical units
obsunits True
# verbose output when running
verbose True
# save the stack data
savedata True
# file path for saving the stack data
savepath stack_bootstrap_output
# for simulations -- to hit a target number of objects exactly
# to split this up by field, pass a list [n_field1, n_field2, n_field3]
goalnumcutouts False

""" plotting parameters """
# save plots to disk
saveplots False
# do the spatial plot
plotspace False
# RADIUS of the output diagnostic spatial images in pixels
# only used if plotspace == True
spacestackwidth 1
 # do the spectral plot
plotfreq False
# TOTAL number of channels to include in the diagnostic spectral images
# only used if plotfreq == True
freqstackwidth 1
# extra diagnostic plots for the cubelet output
plotcubelet False

""" bootstrap-specific parameters """
# number of redshift bins to use for getting the redshift distribution
nzbins 3
# parameters for saving as the bootstrap runs (in case it crashes partway through)
# this is the worst possible way to do this -- need to set up a better iterator
itersave = True
# file for the saves during the run
itersavefile = params.savepath + 'random_stacker_iter_output.csv'
# file for saving at the end of the run
nitersavefile = params.savepath + 'random_stacker_output.npz'
# every N iterations, save
itersavestep = 10

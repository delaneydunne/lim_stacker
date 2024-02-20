# lim_stacker

Code to stack line-intensity mapping (LIM) data cubes on galaxy catalogues in three dimensions. The methodology is explained in detail in [this paper](https://ui.adsabs.harvard.edu/abs/2023arXiv230409832D/abstract). This was developed primarily for the [COMAP](https://comap.caltech.edu/) line-intensity mapping experiment, using [eBOSS](https://www.sdss4.org/surveys/eboss/) or [HETDEX](https://hetdex.org/) as the galaxy catalog on which to stack.

## Running the code

**run_stack.py** contains an example run of the stack.

All of the parameters for adjusting how the stack is run are in **param_defualts.py**, with associated explanations. Parameters that may be useful are:
* *xwidth* / *ywidth* / *freqwidth*: The size of the aperture used to calculate the stack values, in RA/Dec/z pixels respectively. These each default to 3 pixels.
* *centfreq*: The rest frequency of the spectral line being traced by the LIM map. Defaults to CO(1-0), or 115.27 GHz.
* *savepath*: File path for saving the stack data.

Paths to the input maps and catalogs should be adjusted in **run_stack.py** before running.

## Input data formats

The code takes two types of data as inputs: three-dimensional LIM cubes and galaxy catalogs mapping sources in 3D. These should be formatted as:

### LIM Cube

These are loaded and managed by the *maps* class in **tools.py**. Currently, the code can handle either an HDF5 (.h5) or numpy (.npz) file format, and will catch and manage these automatically. The default file type is that output by the [COMAP Pipeline](https://github.com/COMAP-LIM/pipeline/tree/main).

Another file type can be added by specifying a *load* function inside the *maps* class. The input maps should be in brightness temperature units, and should contain RA, Dec, and frequency axes. 

### Galaxy Catalog

These are loaded and managed by the *catalogue* class in **tools.py**. Currently, the code can only handle a numpy (.npz) file format. The catalog should contain at least three files: *ra*, *dec*, and *z* (the redshift). These should each be numpy arrays containing the information for each object.
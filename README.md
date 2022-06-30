# lim_stacker

Code to stack line-intensity mapping data cubes on galaxy catalogues.

**run_stack.py** contains an example run of the stack

**run_bootstraps.py** contains an example run of stacking on random locations in the 3D cube as a consistency check.

TO DO:
- change what cubelet params flag does?
  - cubelet/cubestack bug
- integrate peak-patch simulation combining and stacking into package
- per-field stacks -- create an automatic output flag
- make a custom params class and set it to auto-populate with defaults
- tidy output objects in general -- unzip/print_dict/dict_saver could talk to each other better
- bootstrapping and sim code -- very out-of-date with the cubelet stuff
- weighting issue when params.beamscale is set to True -- returning results that seem to be self-consistent but about a factor of 10 too small
- symmetrized stacking c.f. https://arxiv.org/pdf/2206.03300.pdf 
- subpixel / higher-resolution stacking code
- stack before mapmapking?

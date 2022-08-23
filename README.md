# lim_stacker

Code to stack line-intensity mapping data cubes on galaxy catalogues.

**run_stack.py** contains an example run of the stack

**run_bootstraps.py** contains an example run of stacking on random locations in the 3D cube as a consistency check.

TO DO:
- include a custom number of (good) catalogue objects in the stack
- integrate peak-patch simulation combining and stacking into package
- tidy output objects in general -- unzip/print_dict/dict_saver could talk to each other better
- bootstrapping and sim code -- very out-of-date with the cubelet stuff
- symmetrized stacking c.f. https://arxiv.org/pdf/2206.03300.pdf 
- subpixel / higher-resolution stacking code
- stack before mapmapking?

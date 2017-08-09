# import modules and define directory paths for reduction.py and loadmirdata.py

from matplotlib import pyplot as plt
from IPython.display import Image, display
from IPython.display import Javascript
import numpy as np
import os, glob, subprocess, time, psutil, sys, shutil, fnmatch, argparse
import pandas as pd
from mirpy import miriad 
# note: mirpy is a python2 module that I futurized
# so I could use it in python3
from reduction_funcs import * 



# handle command line arguments
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('band', nargs=1, type=str, choices=['X','C','L'], help='X, C, or L')
parser.add_argument('dir', nargs=1, type=str, help='directory with /raw and /blocks')
parser.add_argument('-sf','--skip_manual_flagging', help='skip flagging manually', action="store_true", default = False)
parser.add_argument('-d','--display_results',  help='don\'t output to terminal', action="store_true", default = False)
args = parser.parse_args()
#parser.print_help()

band = args.band[0]
date_dir = args.dir[0]
display_results = args.display_results

if args.skip_manual_flagging:
    manual_flagging = False
else:
    manual_flagging = True


if band == 'L':
	suffix = '.2100'
	rawfiles = 'L'
	ifsel = 1
elif band == 'C':
	suffix = '.5500'
	rawfiles ='CX'
	ifsel = 1
elif band == 'X':
	suffix = '.9000'
	rawfiles ='CX'
	ifsel = 2


# define directory paths
# where you keep the scripts
notebook_dir = '/Users/mmcintosh/Dropbox/ASTRON2017/C3030/reduction_scripts/'#os.getcwd()
# where you want to save key pngs for quick reference
image_dir = "/Users/mmcintosh/Dropbox/ASTRON2017/C3030/reduction_plots/{0}_{1}_images".format(date_dir,band)
# where your raw data is saved (expects it in files called L or CX)
raw_data_dir = "/Volumes/mjrose/C3030/{0}/raw".format(date_dir)
# where you want to save the visibilities, other pngs, logs
processed_data_dir = "/Volumes/mjrose/C3030/{0}/reduced_{1}".format(date_dir,band)



if not os.path.exists(image_dir):
	os.makedirs(image_dir)

if not os.path.exists(processed_data_dir):
	os.makedirs(processed_data_dir)

# if you need to change the reference antenna, which I needed for '2016'
refant = 4

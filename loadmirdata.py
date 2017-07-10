from reduction_setup import * 


# checking the contents of the directory we're in 
os.chdir(raw_data_dir)
infiles = glob.glob(raw_data_dir+'/*')

# load in the data 
os.chdir(processed_data_dir)
loadeduv = "loaded.uv"

# atlod parameters
in_ = infiles #"2015-04-11_0513.C3030"#"2015-04-11_0149.C3030"#infiles 
out= loadeduv
ifsel= ifsel
options = 'birdie,rfiflag,noauto,xycorr'

# remove uvdata if it already exists
try:
    os.remove(out)
except:
    pass
# store miriad output in variable in case you want to parse it
mir_output = miriad.atlod(in_=in_, out=out, ifsel=ifsel, options=options)

# but let's take a look at it
print(mir_output.decode("utf-8"))


# flag the edge channels that are within the bandpass rollof

# uvflag parameters
vis = loadeduv
edge = 40
flagval = "flag"

mir_output = miriad.uvflag(vis=vis, edge=edge, flagval=flagval)
print(mir_output.decode("utf-8"))


# for this data set I need to order the sources 
# so each can be the phase calibrator for the next


# uvsplit parameters
vis = loadeduv

# remove vis if it already
try:
    shutil.rmtree('*'+suffix)
except:
    pass

mir_output = miriad.uvsplit(vis=vis)
print(mir_output.decode("utf-8"))



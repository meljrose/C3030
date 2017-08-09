from matplotlib import pyplot as plt
from IPython.display import Image, display
from IPython.display import Javascript
import numpy as np
import os, glob, subprocess, time, psutil, sys, shutil,fnmatch, re
import pandas as pd



def recursive_glob(path, regex):
    matches = []
    for root, dirnames, filenames in os.walk(path):
        temp = filenames
        for filename in fnmatch.filter(filenames, regex):
            matches.append(os.path.join(root, filename))
    return(matches)


def find_phasecal(phasecal_df,h):
    # check if phasecal has been already assigned
    if phasecal_df.loc[h]["phasecal"]:
        return(True)
    if h == 0:
        potential_phasecal_name = phasecal_df.loc[h]["name"]
        for i in (phasecal_df[phasecal_df["name"] == phasecal_df.loc[h]["name"]].index.values):
            phasecal_df.set_value(h, 'phasecal', potential_phasecal_name)
        return(True)
    j=1
    while True:
        potential_phasecal_name = phasecal_df.loc[h-j]["name"]
        potential_phasecal_flux = phasecal_df.loc[h-j]["flux"]
        potential_phasecal_percent_flagged = phasecal_df.loc[h-j]["percent_flagged"]
        #print(potential_phasecal_name,potential_phasecal_flux,potential_phasecal_percent_flagged)
        if not potential_phasecal_flux:
            try:
                h_arr = phasecal_df[(phasecal_df.index < (h-j)) & (phasecal_df['flux'] != '')].index[0]
            except:
                # unable to set phase cal if no flux
                return(False)
            temp = np.max(h_arr)
            #print("temp: {0}".format(temp))
            if temp:
                #print("redo reductions from {0}".format(temp))
                return(temp)
            else:
                return(0)
        if float(potential_phasecal_flux) > 1.0 and float(potential_phasecal_percent_flagged) < 50.0:
            for i in (phasecal_df[phasecal_df["name"] == phasecal_df.loc[h]["name"]].index.values):
                phasecal_df.set_value(h, 'phasecal', potential_phasecal_name)
            return(True)
        else:
            if h > j:
                j+=1
            else:
                print("you've run out of phasecals to test")
                return(False)

            
def rm_outfile(outfiles):
    if isinstance(outfiles,list):
        for out in outfiles: 
            try:
                print("trying to remove " + out)
                shutil.rmtree(out)
            except:
                pass            
    else:
        try:
            shutil.rmtree(outfiles)
        except:
            pass

# returns h if not yet reduced or the next h that needs to be reduced

def check_ifreduced(processed_data_dir, sources, h, phasecal_df, suffix, redo=False):
    
    source = sources[h]
    filenames = glob.glob(processed_data_dir+'/*_reduction')
    already_reduced = [file.split('/')[-1].split("_")[0] for file in filenames]
    if source in already_reduced:        
        while source in already_reduced:
            if h < (len(sources)-1):
                
                #print('phasecal', not phasecal_df.loc[h]["flux"], phasecal_df.loc[h]["flux"])
                # if the source files exist but there is no flux listed
                # redo that reduction
                
                if not phasecal_df.loc[h]["flux"]:
                    # remove old reduction files, and return that h to be reduced
                    base = source.split(suffix)[0]
                    file_list = glob.glob(processed_data_dir+"/"+base+'*')
                    
                    if os.path.exists(processed_data_dir+'/'+source):
                        file_list.remove(processed_data_dir+'/'+source)
                        if redo: 
                            rm_outfile(file_list)
                        return(h)
                    else:
                        print("data not found for {0}".format(source))


                h+=1
                source = sources[h]
                while source == '':
                    if h < (len(sources)-1):
                        h+=1
                        source = sources[h]
                
                filenames = glob.glob(processed_data_dir+'/*_reduction')
                already_reduced = [file.split('/')[-1].split("_")[0] for file in filenames]
                
            else:
                print("all sources are already reduced")
                return(None)
    return(h)




# takes mir_output from uvsplit and return the names of the new files
def names_uvsplit(mir_output):
    tmp = mir_output.decode("utf-8", errors="replace").split("\n")[2:]
    temp = [t.split(' ')[-1] for t in tmp]
    return(temp)
    

# takes mir_output from uvsplit and return the names of the new files
def grabrms_invert(mir_output):
    tmp = mir_output.decode("utf-8", errors="replace").split("\n")[2:]
    for k, j in enumerate(tmp):
        if 'rms' in j:
            return(j.split(': ')[-1])


def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()
        

        
        
def log_it(log_name, introduction, mir_output):
    with open(log_name, "a") as myfile:
        myfile.write("-"*10+" {0} ".format(introduction)+"-"*10)
        try:
            myfile.write("\n \n {0} \n \n ".format(mir_output.decode("utf-8", errors="replace")))
        except:
            myfile.write("\n \n {0} \n \n ".format(mir_output))
        
# issue a warning if pgflags more than 50% of the data
def flaggingcheck_pgflag(mir_output):
    tmp = mir_output.decode("utf-8", errors="replace").split("\n")[-1].split('%')[0]
    tmp = float(tmp)
    if tmp >= 50:
        warning = '!'*500 + 'Warning! You are flagging {0}% of the data, which is a lot'.format(tmp)+'!'*500
    else:
        warning=''
    return([tmp, warning])


# check if the rms decreases with each iteration
def rmscheck_clean(mir_output, display_results, log_name=False):
    tmp = mir_output.decode("utf-8", errors="replace").split("\n")[2:]
    
    min_arr = []
    max_arr = []
    rms_arr = []
    iter_arr = []
    flux_arr = []
    for k, j in enumerate(tmp):
        if 'rms' in j:
            temp = j.split(': ')[-1].split('  ')
            min_arr.append(temp[1])
            max_arr.append(temp[2])
            rms_arr.append(temp[3])
        elif 'Iterations' in j:
            iter_arr.append(j.split(': ')[-1])
        elif 'flux' in j:
            flux_arr.append(j.split(': ')[-1])
    
    # check that the rms is monotonically decreasing
    dx= np.diff(np.asarray(rms_arr,dtype=float))
    all_decreasing = np.all(dx <= 0)
    
    if not all_decreasing:
        culprit = iter_arr[np.argmax(dx>0)]
        warning = "Warning: RMS values are not monotonically decreasing after {0} iterations".format(culprit)
        if display_results:
            print(warning)
        if log_name:    
            log_it(log_name,"Warning", warning)
    
    # get a visual 
    '''
    plt.title("Total Cleaned Flux")
    plt.plot(iter_arr, flux_arr)
    plt.xlabel('Iterations')
    plt.ylabel('Flux')
    plt.show()
    '''
    
    plt.title("Residual RMS")
    #plt.plot(iter_arr, min_arr, label="min")
    #plt.plot(iter_arr, max_arr, label="max")
    plt.plot(iter_arr, rms_arr, label="rms")
    
    if not all_decreasing:
        plt.axvline(x=float(culprit), linestyle="--")
                
        
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Iterations')
    plt.ylabel('Flux')
    if display_results:
        plt.show()
    plt.savefig("rms_plot.png")
    plt.close()
    
# grab the peak flux value
# returns [peak value, error]
def grabrms_imhist(mir_output):
    tmp = mir_output.decode("utf-8", errors="replace").split("\n")[2:]
    for k, j in enumerate(tmp):
        if 'rms' in j:
            temp = tmp[k+1].split(' ')
            ret = [t for t in temp if t !='']
            return(ret[2])

def grabflux_uvfmeas(mir_output):
    tmp = mir_output.decode("utf-8", errors="replace").split("\n")[2:]
    for k, j in enumerate(tmp):
        if 'Scalar' in j:
            temp = tmp[k].split(' ')
            ret = [t for t in temp if t !='']
            return(float(ret[3]))


# grab the peak flux value
# returns [peak value, error]
def grabpeak_imfit(mir_output):
    tmp = mir_output.decode("utf-8", errors="replace").split("\n")[2:]
    for k, j in enumerate(tmp):
        if 'Peak' in j:
            temp = j.split(': ')[-1].split(' ')
            ret = [t for t in temp if t !='']
            return(ret[::2])
    
                
                
                
# save as HTML
# I also had to install nbconvert
# conda install nbconvert
# conda install -c conda-forge jupyter_contrib_nbextensions
def export_to_html(notebook_dir,filename,source):
    save_as = notebook_dir+'/'+source+'_reduction.html'
    try:
        os.remove(save_as)
    except:
        pass
    rename_cmd = filename.replace(".ipynb", ".html") + ' ' + save_as
    print(rename_cmd)
    cmd = 'jupyter nbconvert --to html_embed --template toc2 {0}'.format(filename)
    subprocess.call(cmd, shell=True)
    return(save_as)
    # time.sleep(20)
    # if os.path.isfile(filename):
    #     print("File exported as:\n\t{0}".format(fn))
    #     print(time.strftime("%d/%m/%Y"))
    #     return(source+'_reduction.html')
    # else:
    #     # try again
    #     time.sleep(20)
    #     fn = export_to_html(filename,source)

def save_notebook():
    display(Javascript("IPython.notebook.save_notebook()"),
                   include=['application/javascript'])
    return display(Javascript("IPython.notebook.create_checkpoint()"),
                   include=['application/javascript'])

def move_and_display_pngs(device,dir_name,log, display_results):
    png_name = device.split('/png')[0]
    plot_list = glob.glob(png_name+'*')
    for plot in plot_list:
        os.rename(plot,"".join(plot.split(".png"))+('.png'))
    
    plot_list = glob.glob(png_name.split('.png')[0]+'*.png')
    if display_results:
        for plot in plot_list:
            display(Image(filename=plot, format='png'))

    # # rm them if they already exist
    # try:
    #     os.remove(path_list)
    # except:
    #     pass
    
    [os.rename(plot, dir_name+"/"+plot) for plot in plot_list]
    
    if log: 
        os.rename(log,dir_name+"/"+log)
        
         

def sorted_nicely(l): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def check_if_data_unpacked(phasecal_df, processed_data_dir,df_path):
    """ if there are sources in the block but not in the data,
     don't include them in the reduction """
    to_remove = []
    sources = phasecal_df["name"].values.tolist()
    for index, row in phasecal_df.iterrows():
        source = sources[index]
        if not os.path.exists(processed_data_dir+'/'+source):
            to_remove.append(index)
    if to_remove:
        print('missing data for {0} --- removing from reduction list'.format([sources[i] for i in to_remove]))
        phasecal_df.drop(phasecal_df.index[to_remove], inplace=True)
        phasecal_df.reset_index(drop=True, inplace=True)
        phasecal_df.to_csv(df_path)

def atca_filter(n, freq_arr, flux_arr, flux_err_arr):
    # frequency filter for plotting
    mask = np.ones(len(freq_arr), dtype=bool)
    ind = np.where((freq_arr >= 1290) & (freq_arr <= 3030))[0]
    ind = np.append(ind, np.where((freq_arr >= 4500) & (freq_arr <= 6430))[0]) 
    ind = np.append(ind,np.where((freq_arr >= 8070) & (freq_arr <= 9930))[0])
    indmask = np.ones(len(ind), dtype=bool) 
    indmask[::n] = False
    ind = ind[indmask]
    mask[ind] = False
    return(freq_arr[mask], flux_arr[mask], flux_err_arr[mask])


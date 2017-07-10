from reduction_setup import * 


# dataframe for the source being processed to find its best flux calibrator
df_path = processed_data_dir+'/calibration_order.csv'

if not os.path.exists(df_path):

    # read in list of souces from mosaic order
    blocks = np.arange(1,11,1)
    temp_arr = []
    
    block_dir = raw_data_dir.split('/raw')[0]+"/blocks"
    # get list of blocks
    block_files = glob.glob(block_dir+"/*")
    block_files = sorted_nicely(block_files)
    
    # remove the bandflux calibrator file
    bfcal_path = [a for a in block_files if 'bandfluxcal' in a][0]
    block_files = [a for a in block_files if 'bandfluxcal' not in a]
    for block in block_files: 
        temp_txt = np.loadtxt(block, dtype=bytes)
        temp_arr.append([t[-1].decode('UTF-8')[1:].lower() for t in temp_txt])

    # get the unique blocks if any are repeated
    unique_blocks = np.sort(np.unique(temp_arr, return_index=True)[1])
    sources = np.concatenate([temp_arr[index] for index in unique_blocks])
    sources = [s+suffix for s in sources]
    # specify the seperate bandflux cal
    
    bandflux_cal = str(np.genfromtxt(bfcal_path, dtype=str))+suffix
    sources = np.append(bandflux_cal,sources)
    print(sources)

    # some are repeated but they are in the order we need for phase calibration

    # init a dataframe to keep track of phasecals
    df_init = pd.DataFrame({'name' : sources, 'percent_flagged': '', 'flux' : '', 'phasecal':''})
    df_init.to_csv(df_path)




# get list of sources
phasecal_df = pd.DataFrame.from_csv(df_path)
phasecal_df = phasecal_df.fillna('')


sources = phasecal_df["name"].values.tolist()
bandflux_cal = sources[0]



os.chdir(processed_data_dir)
# iteratively reduce the data
for h in phasecal_df.index.values.tolist():

# guess where to start


    # if you never got flux from your last reduction, go redo that one
    while True: 
        h = check_ifreduced(processed_data_dir, sources, h, phasecal_df, suffix)
        print(h)
        if h is None:
            print("all reduced")
            exit()
            break
        p = find_phasecal(phasecal_df,h)
        if isinstance(p, bool):
            break
        else:
            h = p


    print(h,'/',len(sources))
    source = sources[h]
    print('source = '+source) 
    print('phasecal = '+phasecal_df.loc[h]["phasecal"])


    # make directory to store pngs, logs from reduction
    dir_name = source +"_reduction"
    log_name = dir_name+"/miriadoutput.txt"

    # remove if it already exists

    rm_outfile(dir_name)

    os.makedirs(dir_name)
    log_it(log_name,"Miriad Output for {0} on {1} \n \n".format(source,time.strftime("%d/%m/%Y")), "")
    
    
    
     
    if source == bandflux_cal:
        # get an idea of what the data looks like before calibration
        if display_results:
            print('#'*10+"Pre mfcal bandpass, flux calibration"+'#'*10)

        # uvspec parameters
        vis = source
        stokes = "xx,yy"
        axis = "chan,amp"
        device="{0}_premfcal.png/png".format(vis)
        mir_output = miriad.uvspec(vis=vis,stokes=stokes,axis=axis,
                                    device=device)

        move_and_display_pngs(device,dir_name,log=None, display_results=display_results)

        # determine the bandpass shape

        # mfcal parameters
        vis = source
        mir_output = miriad.mfcal(vis=vis)
        if display_results:
            print(mir_output.decode("utf-8"))

        # if you want to see what it looks like after mfcal
        if display_results:
            print('#'*10+"Post mfcal bandpass, flux calibration"+'#'*10)
        # uvspec parameters
        vis = source
        stokes = "xx,yy"
        axis = "chan,amp"
        device="{0}_postmfcal.png/png".format(vis)
        mir_output = miriad.uvspec(vis=vis,stokes=stokes,axis=axis,
                                    device=device)

        move_and_display_pngs(device,dir_name,log=None, display_results=display_results)


    else:   
        
        # copy over the calibration solution

        # gpcopy parameters
        vis = phasecal_df.loc[h]["phasecal"]
        out = source
        mir_output = miriad.gpcopy(vis=vis,out=out)
        if display_results:
            print(mir_output.decode("utf-8"))
        




    # uvfmeas!
    vis= source
    stokes ='i'
    line = 'channel,2048,1,1,1'
    log = 'precaluvfmeaslog{0}MHz_{1}'.format(line.split(',')[-2],source)
    device="{0}_precaluvmeas.png/png".format(source)
    feval="2.1"

    mir_output = miriad.uvfmeas(vis=vis,stokes=stokes,device=device,line=line,
                                feval=feval,log=log)
    log_it(log_name,"uvfmeas", mir_output)

    # put it in notebook dir
    im_to_save = "{0}_precaluvmeas.png".format(source)
    shutil.copy(im_to_save, image_dir)

    if display_results:
        print(mir_output.decode("utf-8", errors='replace'))

    move_and_display_pngs(device,dir_name,log, display_results=display_results)



    #################### RFI FLAGGING #####################
    

    # do this 2 or 3 times
    if source == bandflux_cal:
        loop = [0,1,2]
    else:
        loop = [0,1]

    for l in loop: 

        # RFI FLAGGING

        # loop over stokes
        stokes_arr = ['xx','xy','yx','yy']

        for u in stokes_arr:
            # pgflag parameters
            vis = source
            stokes = str(u)+','+','.join(stokes_arr)
            command = "<b"
            device="/xs" 
            options="nodisp"
            mir_output = miriad.pgflag(vis=vis,stokes=stokes,command=command,
                                options=options, device=device)
            flagging_check = flaggingcheck_pgflag(mir_output)
            log_it(log_name,"pgflag", mir_output)
            # add flagging_check[0] to df
            if display_results:
                print(mir_output.decode("utf-8", errors="replace"))


        if manual_flagging:
            # blflag parameters
            vis = source
            device="/xs" 
            stokes = "xx,yy"
            axis = "chan,amp"
            options="nofqav,nobase,selgen"
            mir_output = miriad.blflag(vis=vis,device=device,stokes=stokes,
                                       axis=axis,options=options)
            if display_results:
                print(mir_output.decode("utf-8", errors="replace"))

            log_it(log_name,"blflag", mir_output)

            # save flagging log
            log = dir_name+"/blflag.select{0}".format(l)
            try:
                shutil.move('blflag.select', log)
            except:
                pass


    for i in (phasecal_df[phasecal_df["name"] == phasecal_df.loc[h]["name"]].index.values):    
        phasecal_df.set_value(i, 'percent_flagged', flagging_check[0])    
    # if you want to see what it looks like after blflag
    
    # uvspec parameters
    if display_results:
        print('#'*10+"Post blflag source"+'#'*10)
    vis = source
    stokes = "xx,yy"
    axis = "chan,amp"
    #device="/xs"
    device="{0}_postblflag.png/png".format(vis)
    mir_output = miriad.uvspec(vis=vis,stokes=stokes,axis=axis,
                                device=device)

    if display_results:
        print(mir_output.decode("utf-8", errors="replace"))


    move_and_display_pngs(device,dir_name,log=None, display_results=display_results)

    # done RFI flagging the source
    # now apply the calibration solutions

    
    # look at the amplitude vs time
    if display_results:
        print('#'*10+"Check if source is resolved"+'#'*10)
    # uvplt
    vis = source
    stokes = "xx,yy"
    axis = "uvdistance,amplitude"
    options="nobase"
    device="{0}_uvdistamp.png/png".format(vis)

    mir_output = miriad.uvplt(vis=vis,stokes=stokes,axis=axis,
                              options=options,device=device)
    if display_results:
        print(mir_output.decode("utf-8"))

    move_and_display_pngs(device,dir_name,log=None, display_results=display_results)
    
    
    
    #################### AFTER RFI FLAGGING #####################


    # gpcal parameters
    vis =  source
    interval="0.1"
    nfbin="4"
    options="xyvary"

    mir_output = miriad.gpcal(vis=vis,interval=interval,nfbin=nfbin,
                               options=options)

    log_it(log_name,"gpcal", mir_output)

    if display_results:
        print(mir_output.decode("utf-8"))


    # uvplt
    vis = source
    stokes = "xx,yy"
    axis = "real,imag"
    options="nofqav,nobase,equal"
    device="{0}_realimag.png/png".format(vis)

    mir_output = miriad.uvplt(vis=vis,stokes=stokes,axis=axis,
                              options=options,device=device)
    if display_results:
        print(mir_output.decode("utf-8"))

    move_and_display_pngs(device,dir_name,log=None, display_results=display_results)


    if not (source == bandflux_cal):
        # reestablish flux
        # gpboot parameters
        vis = source
        cal = bandflux_cal

        mir_output = miriad.gpboot(vis=vis,cal=cal)
        log_it(log_name,"gpboot", mir_output)

        if display_results:
            print(mir_output.decode("utf-8"))

    # uvfmeas!
    vis= source
    stokes ='i'
    line = 'channel,2048,1,1,1'
    log = 'uvfmeaslog{0}MHz_{1}'.format(line.split(',')[-2],source)
    device="{0}_postcaluvmeas.png/png".format(source)
    feval="2.1"

    mir_output = miriad.uvfmeas(vis=vis,stokes=stokes,device=device,line=line,
                                feval=feval,log=log)
    log_it(log_name,"uvfmeas", mir_output)

    # put it in notebook dir
    im_to_save = "{0}_postcaluvmeas.png".format(source)
    shutil.copy(im_to_save, image_dir)

    if display_results:
        print(mir_output.decode("utf-8", errors='replace'))

    move_and_display_pngs(device,dir_name,log, display_results=display_results)


    flux_uvfmeas = grabflux_uvfmeas(mir_output) 
    for i in (phasecal_df[phasecal_df["name"] == phasecal_df.loc[h]["name"]].index.values):    
        phasecal_df.set_value(i, 'flux', flux_uvfmeas)  
    phasecal_df.to_csv(df_path)
    
    
        #################### IMAGING #####################

    if source is not bandflux_cal:
        # average/smoothe antenna gains
        # gpaver parameters
        vis = source
        interval = "2"

        mir_output = miriad.gpaver(vis=vis,interval=interval)
        if display_results:
            print(mir_output.decode("utf-8"))


        # uvaver parameters
        vis = source
        temp = vis.split('.')
        uvav_source = temp[0]+'.uvaver.'+temp[1]
        out= uvav_source 

        # remove it if it already exists
        # remove if it already exists
        rm_outfile(out)

        mir_output = miriad.uvaver(vis=vis,out=out)
        if display_results:
            print(mir_output.decode("utf-8"))

        # split into smaller frequency chunks

        # uvsplit parameters
        vis = uvav_source 
        maxwidth ="0.512"

        # how do I predict these file names? 
        splitname = source.split('.2100')[0]
        endings = ['.2868','.2356','.1844','.1332']
        # remove if it already exists
        rm_outfile([splitname+end for end in endings])

        mir_output = miriad.uvsplit(vis=vis,maxwidth=maxwidth)
        if display_results:
            print(mir_output.decode("utf-8"))

        freq_chunks = names_uvsplit(mir_output)


        # Imaging

        # select the first freq chunk
        freq_chunk = freq_chunks[0]

        # do I need to image each frequency chunk? 
        #for freq_chunk in freq_chunks:
        #    print(freq_chunk)

        # loop twice so you can use self-cal
        loop = [0,1]

        for l in loop:

            if l == 1:
                ##########################################################################
                # self calibration
                if display_results:
                    print('#'*10+"Selfcal"+"#"*10)
                # selfcal parameters
                vis = freq_chunk
                model=vis+".imodel"
                clip= "0.013"
                interval=1
                options="phase,mfs"

                try: 
                    mir_output = miriad.selfcal(vis=vis,model=model,clip=clip,interval=interval,
                                               options=options)
                    if display_results:
                        print(mir_output.decode("utf-8"))
                except:
                    print("couldn't self-cal")
                ###########################################################################


            # invert parameters
            vis = freq_chunk
            map_ = freq_chunk+".imap"
            beam = freq_chunk+".ibeam"
            robust = "0.5"
            stokes = "i"
            options="mfs,double"

            # delete the files if they already exist
            rm_outfile(map_)
            rm_outfile(beam)

            mir_output = miriad.invert(vis=vis,map=map_,beam=beam,robust=robust,
                                       stokes=stokes, options=options)
            if display_results:
                print(mir_output.decode("utf-8"))


            invert_rms = grabrms_invert(mir_output)

            # look at the dirty map

        # dirty map
            # cgdisp parameters
            in_ = freq_chunk+".imap"
            beam = freq_chunk+".ibeam"
            type_="p"
            device="{0}_dirtymap.png/png".format(in_)
            laptyp = "/hms,dms"
            options="wedge"

            mir_output = miriad.cgdisp(in_=in_,beam=beam,type=type_,device=device,laptyp=laptyp,
                                       options=options)
            if display_results:
                print(mir_output.decode("utf-8"))
                print('#'*10+"Dirty Map after inversion"+"#"*10)
            move_and_display_pngs(device,dir_name,log=None,display_results=display_results)


        # Dirty beam
            # cgdisp parameters
            in_= freq_chunk+".ibeam"
            type_="p"
            device="{0}_dirtybeam.png/png".format(in_)
            laptyp = "/hms,dms"
            options="wedge"
            range_="0,0,log"

            mir_output = miriad.cgdisp(in_=in_,type=type_,device=device,laptyp=laptyp,
                                       options=options,range=range_)
            if display_results:
                print(mir_output.decode("utf-8"))
                print('#'*10+"beam"+"#"*10)
            move_and_display_pngs(device,dir_name,log=None,display_results=display_results)



            # clean parameters
            map_ = freq_chunk+".imap"
            beam = freq_chunk+".ibeam"
            out = freq_chunk+".imodel"
            options="negstop,positive"
            cutoff=invert_rms
            niters="1000"

            # remove model if it already exists
            rm_outfile(out)

            mir_output = miriad.clean(map=map_,beam=beam,out=out,cutoff=cutoff,niters=niters,
                                       options=options)
            if display_results:
                print(mir_output.decode("utf-8"))


            rmscheck_clean(mir_output, display_results,log_name) 

            # I should go back and make these plot better

            # restore the image

            # restor parameters
            model = freq_chunk+".imodel"
            beam = freq_chunk+".ibeam"
            map_ = freq_chunk+".imap"
            out = freq_chunk+".irestor"
            mode=""#"residual"

            # remove restored image if it already exists
            rm_outfile(out)

            mir_output = miriad.restor(model=model,map=map_,beam=beam,out=out, mode=mode)
            if display_results:
                print(mir_output.decode("utf-8"))

            # look at the cleaned image



            # cgdisp parameters
            in_ = freq_chunk+".irestor"
            beam = freq_chunk+".ibeam"
            type_="p"
            device="{0}_cleanedimage.png/png".format(in_)
            laptyp = "/hms,dms"
            options="wedge"
            range_="0,0,log"

            mir_output = miriad.cgdisp(in_=in_,beam=beam,type=type_,device=device,laptyp=laptyp,
                                       options=options,range=range_)

            # put it in notebook dir
            im_to_save = '{0}_cleanedimage.png'.format(in_)
            shutil.copy(im_to_save, image_dir)

            if display_results:
                print(mir_output.decode("utf-8"))
                print('#'*10+"Cleaned image"+"#"*10)
            move_and_display_pngs(device,dir_name,log=None, display_results=display_results)




            # measure the flux density of the source

            # imfit parameters
            in_ = freq_chunk+".irestor"
            region="quarter"
            object_="point"
            spar="1,0,0"
            out=freq_chunk+".iresidual"
            options="residual"

            # remove residual image if it already exists
            rm_outfile(out)

            mir_output = miriad.imfit(in_=in_,region=region,object=object_,spar=spar, out=out,
                                       options=options)
            if display_results:
                    print(mir_output.decode("utf-8"))

            peakflux= grabpeak_imfit(mir_output)



            # look at the cleaned image after the central source is subtracted by imfit



            # cgdisp parameters
            in_ = freq_chunk+".iresidual"
            beam = freq_chunk+".ibeam"
            type_="p"
            device="{0}_cleanedsubtracted.png/png".format(in_)
            laptyp = "/hms,dms"
            options="wedge"
            range_="0,0,log"

            mir_output = miriad.cgdisp(in_=in_,beam=beam,type=type_,device=device,laptyp=laptyp,
                                       options=options,range=range_)
            if display_results:
                print(mir_output.decode("utf-8"))
                print('#'*10+"Cleaned image - central source"+"#"*10)
            move_and_display_pngs(device,dir_name,log=None, display_results=display_results)


            # estimate the dynamic range of the image

            # imhist parameters
            in_ = freq_chunk+".iresidual"
            region="quarter"
            device="{0}_imhist.png/png".format(in_)
            options="nbin,100"

            mir_output = miriad.imhist(in_=in_,region=region, device=device,
                                       options=options)
            if display_results:
                print(mir_output.decode("utf-8"))
                print('#'*10+"imhist"+"#"*10)
            move_and_display_pngs(device,dir_name,log=None, display_results=display_results)
            log_it(log_name,"imhist", mir_output)


            rms_imhist= grabrms_imhist(mir_output)

            dynamic_range = float(peakflux[0])/float(rms_imhist)
            log_it(log_name,"dynamic range", dynamic_range)
            if display_results:
                print('dynamic range: {0}'.format(round(dynamic_range)))
            
            







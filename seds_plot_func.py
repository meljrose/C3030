import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from collections import OrderedDict
import os
import re
from matplotlib import rc # To plot labels in serif rather than default.
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['serif'],'size':18})

cdict_models = {'singhomobremss':'red',
        'singhomobremsscurve':'maroon',
        'singhomobremssbreak': 'orangered',
        'singinhomobremss':'k',
        'singinhomobremsscurve':'#4682b4',
        'doubhomobremss':'saddlebrown',
        'doubhomobremsscurve':'dodgerblue',
        'doubhomobremssbreak':'olive',
        'doubhomobremssbreak':'DarkGoldenRod',
        'singSSA':'orchid',
        'singSSAcurve':'darkmagenta',
        'singSSAbreak':'indigo',
        'doubSSA':'navy',
        'doubSSAcurve':'sienna',
        'doubSSAbreak':'black',
        'powlaw': 'DarkOrange',
        'powlawbreak':'Chocolate',
        'internalbremss':'MediumSeaGreen',
        'curve':'k'
            } 

# Defining plotting routine

def sed(models,paras,freq,flux,flux_err, name,
        grid = False, freq_labels = False, log = True, bayes = False, resid = False, savefig=False):
    
    # Ensuring that freq and flux are approability matched up.
    ind = np.argsort(freq)
    freq = freq[ind]
    flux = flux[ind]
    flux_err = flux_err[ind]
    
    if resid == True:
        fig = plt.figure(1,figsize=(15, 10)) #(12,8)
        gs = plt.GridSpec(2,1, height_ratios = [3,1], hspace = 0)
        ax = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        ax1.set_xlabel('Frequency (MHz)', fontsize = 20)
        ax1.set_ylabel(r'$\chi$', fontsize = 20)
        ax1.xaxis.labelpad = 15
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
            ax1.spines[axis].set_linewidth(2)
    else:
        # fig = plt.figure(1,figsize=(12,8))#(12, 8))
        # gs = plt.GridSpec(1,1)
        # ax = plt.subplot(gs[0])

        fig = plt.figure(1,figsize=(15, 10))#(12, 8))
        gs = plt.GridSpec(1,1)
        ax = plt.subplot(gs[0])
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)

        # Make inset plot
        # gs = plt.GridSpec(7,9)
        # fig = plt.figure(1,figsize=(12,8))#(12, 8)) 
        # ax = fig.add_subplot(gs[:, :])
        # ax1 = fig.add_subplot(gs[1, 7]) # top right, 2,0 is bottom left

    freq_cont = np.array(range(60,22000))
    if max(freq) < 1500.:
        ax.set_xlim(70., 1500.)
    else:
        ax.set_xlim(70., 21000.)
    ax.set_ylim(min(flux)-0.1*min(flux), max(flux)+0.2*max(flux))
    ax.set_xlabel('Frequency (MHz)', fontsize = 35,labelpad=10)
    ax.set_ylabel('Flux Density (Jy)', fontsize = 35)
    ax.set_title(name, fontsize = 35)

    try:
        tt = len(models)
    except TypeError:
        tt = 1
        models = [models]
        paras = [paras]

    for i in range(tt):

        # Defining colours for models to make it easy to tell the difference
        try:
            color = cdict_models[models[i].__name__] # In case none of the models are listed here.        
        except KeyError:
            # print 'Model is not in colour dictionary. Defaulting to dark orange.'
            color = 'darkgreen'

        ax.plot(freq_cont, models[i](freq_cont, *paras[i]), color = color, linestyle='-', lw=2)
        
        if resid == True:
            model_points = models[i](freq,*paras[i])
            residuals = flux-model_points
            chi_sing = residuals/flux_err
            chi_sing_err = np.ones(len(freq)) # Chi errors are 1.
            compx = np.linspace(min(freq)-0.1*min(freq),max(freq)+0.1*max(freq))
            compy = np.zeros(len(compx))
            ax1.plot(compx,compy,linestyle = '--',color = 'gray',linewidth = 2)
            ax1.set_xlim(min(freq_cont), max(freq_cont))
            # ax1.errorbar(freq,residuals,flux_err,color = color, linestyle='none',marker = '.')
            for i in range(len(freq)):

                if freq[i] in [232,248,270,280,296,312,328,344,360,376,392,408.01,424,440,456,472]:
                    (_, caps, _) = ax.errorbar(freq[i], flux[i], flux_err[i], marker = 's',elinewidth=2, markersize=8,  color = 'dodgerblue', linestyle='none', label = 'P band',markeredgecolor='none')
                    for cap in caps:
                        cap.set_color('dodgerblue') 
                        cap.set_markeredgewidth(2)
                elif freq[i] in [76,84,92,99,107,115,123,130,143,151,158,166,174,181,189,197,204,212,219,227]:
                    (_, caps, _) = ax.errorbar(freq[i], flux[i], flux_err[i],marker = 'o',elinewidth=2, markersize=10,  color = 'crimson', linestyle='none', label = 'GLEAM',markeredgecolor='none')
                    for cap in caps:
                        cap.set_color('crimson') 
                        cap.set_markeredgewidth(2)                
                elif freq[i] in [74]:
                    (_, caps, _) = ax.errorbar(freq[i], flux[i], flux_err[i], marker = '^',elinewidth=2, markersize=10,  color = 'DarkOrchid', linestyle='none', label = 'VLSSr',markeredgecolor='none')
                    for cap in caps:
                        cap.set_color('DarkOrchid') 
                        cap.set_markeredgewidth(2)
                elif freq[i] in [148]:
                    (_, caps, _) = ax.errorbar(freq[i], flux[i], flux_err[i], marker = 's',elinewidth=2,  markersize=10, color = 'black', linestyle='none', label = 'TGSS',markeredgecolor='none')
                    for cap in caps:
                        cap.set_color('black') 
                        cap.set_markeredgewidth(2)    
                elif freq[i] in [408]:
                    (_, caps, _) = ax.errorbar(freq[i], flux[i], flux_err[i], marker = '<',elinewidth=2, markersize=10,  color = 'forestgreen', linestyle='none', label = 'MRC',markeredgecolor='none')
                    for cap in caps:
                        cap.set_color('forestgreen') 
                        cap.set_markeredgewidth(2)
                elif freq[i] in [843]:
                    (_, caps, _) = ax.errorbar(freq[i], flux[i], flux_err[i], marker = '>',elinewidth=2, markersize=10,  color = 'saddlebrown', linestyle='none', label = 'SUMSS',markeredgecolor='none')
                    for cap in caps:
                        cap.set_color('saddlebrown') 
                        cap.set_markeredgewidth(2)
                elif freq[i] in [1400]:
                    (_, caps, _) = ax.errorbar(freq[i], flux[i], flux_err[i], marker = 'v',elinewidth=2, markersize=10,  color = 'navy', linestyle='none', label = 'NVSS',markeredgecolor='none')
                    for cap in caps:
                        cap.set_color('navy') 
                        cap.set_markeredgewidth(2)
                elif freq[i] in [4800,4850,8640,20000]:
                    (_, caps, _) = ax.errorbar(freq[i], flux[i], flux_err[i], marker = 'd',elinewidth=2, markersize=10,  color = 'darkgreen', linestyle='none', label = 'ATPMN',markeredgecolor='none')
                    for cap in caps:
                        cap.set_color('darkgreen') 
                        cap.set_markeredgewidth(2)
        
                elif 1290 <= freq[i] <= 3030:
                    (_, caps, _) = ax.errorbar(freq[i], flux[i], flux_err[i], marker = 's',elinewidth=2, markersize=8,  color = 'darkmagenta', linestyle='none', label = 'ATCA L',markeredgecolor='none')
                    for cap in caps:
                        cap.set_color('darkmagenta') 
                        cap.set_markeredgewidth(2)            
                elif 4500 <= freq[i] <= 6430:
                    (_, caps, _) = ax.errorbar(freq[i], flux[i], flux_err[i], marker = '^',elinewidth=2, markersize=10,  color = 'indigo', linestyle='none', label = 'ATCA C',markeredgecolor='none')
                    for cap in caps:
                        cap.set_color('indigo') 
                        cap.set_markeredgewidth(2)
                elif 8070 <= freq[i] <= 9930:
                    (_, caps, _) = ax.errorbar(freq[i], flux[i], flux_err[i], marker = 's',elinewidth=2,  markersize=10, color = 'DarkGoldenRod', linestyle='none', label = 'ATCA X',markeredgecolor='none')
                    for cap in caps:
                        cap.set_color('DarkGoldenRod') 
                        cap.set_markeredgewidth(2)    
                else:
                    ax.errorbar(freq[i], flux[i], flux_err[i], marker = 'o',elinewidth=2, markersize=10, color = 'MediumSeaGreen', linestyle='none', label = 'Data',markeredgecolor='none')
                # Elimanating doubled up legend values.
                handles, labels = ax.get_legend_handles_labels()
                by_label = OrderedDict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(),loc='lower center', fontsize=15)


        
            # ax1.errorbar(freq,chi_sing,chi_sing_err,color = color, linestyle='none',marker = '.')
            # ax1.set_ylim(min(chi_sing)-0.2*min(chi_sing), max(chi_sing)+0.2*max(chi_sing))

        
        if log == True:
            ax.set_xscale('log')
            ax.set_yscale('log')
            
            # Making sure minor ticks are marked properly.

            def ticks_format_x(value, index):
                """
                get the value and returns the value as:
                   integer: [0,99]
                   1 digit float: [0.1, 0.99]
                   n*10^m: otherwise
                To have all the number of the same size they are all returned as latex strings
                """
                exp = np.floor(np.log10(value))
                base = value/10**exp
                if value == 9.0 or value == 90. or value == 900. or value == 700.: # This will remove 90 and 900 MHz, replace number for anything you don't want to appear.
                    return ''
                if exp == 0 or exp == 1 or exp == 2 or exp ==3:   
                    return '${0:d}$'.format(int(value))
                if exp == -1:
                    return '${0:.1f}$'.format(value)
                else:
                    return '${0:d}\\times10^{{{1:d}}}$'.format(int(base), int(exp))

            def ticks_format_x_high(value, index):
                """
                get the value and returns the value as:
                   integer: [0,99]
                   1 digit float: [0.1, 0.99]
                   n*10^m: otherwise
                To have all the number of the same size they are all returned as latex strings
                """
                exp = np.floor(np.log10(value))
                base = value/10**exp
                if value in [9.0, 90., 900.,700. ,80,200,300,400,600,700,800,2000,3000,4000,6000,7000,8000,9000,20000]: # This will remove 90 and 900 MHz, replace number for anything you don't want to appear.
                    return ''
                if exp == 0 or exp == 1 or exp == 2 or exp ==3 or exp ==4:   
                    return '${0:d}$'.format(int(value))
                if exp == -1:
                    return '${0:.1f}$'.format(value)
                else:
                    return '${0:d}\\times10^{{{1:d}}}$'.format(int(base), int(exp))

            def ticks_format_y_high(value, index):
                """
                get the value and returns the value as:
                   integer: [0,99]
                   1 digit float: [0.1, 0.99]
                   n*10^m: otherwise
                To have all the number of the same size they are all returned as latex strings
                """
                exp = np.floor(np.log10(value))
                base = value/10**exp
                if value in np.array([0.03,0.05,0.07,0.09,0.3,0.5,0.7,0.9,3,5,7,9,13,15,17,19,23,25,27,29]): # This will remove 90 and 900 MHz, replace number for anything you don't want to appear.
                    # print value
                    # print np.where(value == 0.5)
                    return ''
                # if (value > 0.3) and (value < 0.3001):
                #     print value
                #     print type(value)
                #     print float(value)
                if exp == 0 or exp == 1 or exp == 2 or exp ==3 or exp ==4:   
                    return '${0:d}$'.format(int(value))
                if exp == -1:
                    return '${0:.1f}$'.format(value)
                if exp == -2:
                    return '${0:.2f}$'.format(value)
                else:
                    return '${0:d}\\times10^{{{1:d}}}$'.format(int(base), int(exp))


            def ticks_format_y(value, index):
                """
                get the value and returns the value as:
                   integer: [0,99]
                   1 digit float: [0.1, 0.99]
                   n*10^m: otherwise
                To have all the number of the same size they are all returned as latex strings
                """
                if value in np.array([0.03,0.05,0.07,0.09,0.3,0.5,0.7,0.9,3,5,7,9,13,15,17,19,23,25,27,29]): # This will remove 90 and 900 MHz, replace number for anything you don't want to appear.
                    # print value
                    # print np.where(value == 0.5)
                    return ''
                exp = np.floor(np.log10(value))
                base = value/10**exp
                if exp == 0 or exp == 1 or exp == 2 or exp ==3:   
                    return '${0:d}$'.format(int(value))
                if exp == -1:
                    return '${0:.1f}$'.format(value)
                if exp == -2:
                    return '${0:.2f}$'.format(value)
                else:
                    return '${0:d}\\times10^{{{1:d}}}$'.format(int(base), int(exp))

            subsx = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] # ticks to show per decade
            subsy = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] 
            ax.xaxis.set_minor_locator(ticker.LogLocator(subs=subsx)) #set the ticks position
            ax.xaxis.set_major_formatter(ticker.NullFormatter())   # remove the major ticks
            if max(freq) < 2000:
                ax.xaxis.set_minor_formatter(ticker.FuncFormatter(ticks_format_x))
            else:
                ax.xaxis.set_minor_formatter(ticker.FuncFormatter(ticks_format_x_high))
            ax.yaxis.set_minor_locator(ticker.LogLocator(subs=subsy))
            ax.yaxis.set_major_formatter(ticker.NullFormatter())   # remove the major ticks
            if max(freq) < 2000:
                ax.yaxis.set_minor_formatter(ticker.FuncFormatter(ticks_format_y))
            else:
                ax.yaxis.set_minor_formatter(ticker.FuncFormatter(ticks_format_y_high))

            ax.tick_params(axis='both',which='both',labelsize=22)
            ax.tick_params(axis='both',which='major',length=8,width=1.5)
            ax.tick_params(axis='both',which='minor',length=5,width=1.5)

            if resid == True:
                ax.set_xticklabels('',minor = True)
                ax.set_xlabel('')
                ax1.set_xscale('log')
                ax1.xaxis.set_minor_locator(ticker.LogLocator(subs=subsx)) #set the ticks position
                ax1.xaxis.set_major_formatter(ticker.NullFormatter())   # remove the major ticks
                ax1.xaxis.set_minor_formatter(ticker.FuncFormatter(ticks_format_x))
                ax1.tick_params(axis='both',which='both',labelsize=22)
                ax1.tick_params(axis='both',which='major',length=8,width=1.5)
                ax1.tick_params(axis='both',which='minor',length=5,width=1.5)

    if freq_labels == True: 

        for i in range(len(freq)):

            if freq[i] in [232,248,270,280,296,312,328,344,360,376,392,408.01,424,440,456,472]:
                (_, caps, _) = ax.errorbar(freq[i], flux[i], flux_err[i], marker = 's',elinewidth=2, markersize=8,  color = 'dodgerblue', linestyle='none', label = 'P band',markeredgecolor='none')
                for cap in caps:
                    cap.set_color('dodgerblue') 
                    cap.set_markeredgewidth(2)
            elif freq[i] in [76,84,92,99,107,115,122,123,130,143,151,158,166,174,181,189,197,204,212,219,220,227]:
                (_, caps, _) = ax.errorbar(freq[i], flux[i], flux_err[i],marker = 'o',elinewidth=2, markersize=10,  color = 'crimson', linestyle='none', label = 'GLEAM',markeredgecolor='none')
                for cap in caps:
                    cap.set_color('crimson') 
                    cap.set_markeredgewidth(2)                
            elif freq[i] in [74]:
                (_, caps, _) = ax.errorbar(freq[i], flux[i], flux_err[i], marker = '^',elinewidth=2, markersize=10,  color = 'DarkOrchid', linestyle='none', label = 'VLSSr',markeredgecolor='none')
                for cap in caps:
                    cap.set_color('DarkOrchid') 
                    cap.set_markeredgewidth(2)
            elif freq[i] in [148]:
                (_, caps, _) = ax.errorbar(freq[i], flux[i], flux_err[i], marker = 's',elinewidth=2,  markersize=10, color = 'black', linestyle='none', label = 'TGSS',markeredgecolor='none')
                for cap in caps:
                    cap.set_color('black') 
                    cap.set_markeredgewidth(2)    
            elif freq[i] in [408]:
                (_, caps, _) = ax.errorbar(freq[i], flux[i], flux_err[i], marker = '<',elinewidth=2, markersize=10,  color = 'forestgreen', linestyle='none', label = 'MRC',markeredgecolor='none')
                for cap in caps:
                    cap.set_color('forestgreen') 
                    cap.set_markeredgewidth(2)
            elif freq[i] in [843]:
                (_, caps, _) = ax.errorbar(freq[i], flux[i], flux_err[i], marker = '>',elinewidth=2, markersize=10,  color = 'saddlebrown', linestyle='none', label = 'SUMSS',markeredgecolor='none')
                for cap in caps:
                    cap.set_color('saddlebrown') 
                    cap.set_markeredgewidth(2)
            elif freq[i] in [1400]:
                (_, caps, _) = ax.errorbar(freq[i], flux[i], flux_err[i], marker = 'v',elinewidth=2, markersize=10,  color = 'navy', linestyle='none', label = 'NVSS',markeredgecolor='none')
                for cap in caps:
                    cap.set_color('navy') 
                    cap.set_markeredgewidth(2)
            elif freq[i] in [4800,4850,8640,20000]:
                (_, caps, _) = ax.errorbar(freq[i], flux[i], flux_err[i], marker = 'd',elinewidth=2, markersize=10,  color = 'darkgreen', linestyle='none', label = 'ATPMN',markeredgecolor='none')
                for cap in caps:
                    cap.set_color('darkgreen') 
                    cap.set_markeredgewidth(2)
    
            elif 1290 <= freq[i] <= 3030:
                (_, caps, _) = ax.errorbar(freq[i], flux[i], flux_err[i], marker = 's',elinewidth=2, markersize=8,  color = 'darkmagenta', linestyle='none', label = 'ATCA L',markeredgecolor='none')
                for cap in caps:
                    cap.set_color('darkmagenta') 
                    cap.set_markeredgewidth(2)            
            elif 4500 <= freq[i] <= 6430:
                (_, caps, _) = ax.errorbar(freq[i], flux[i], flux_err[i], marker = '^',elinewidth=2, markersize=10,  color = 'indigo', linestyle='none', label = 'ATCA C',markeredgecolor='none')
                for cap in caps:
                    cap.set_color('indigo') 
                    cap.set_markeredgewidth(2)
            elif 8070 <= freq[i] <= 9930:
                (_, caps, _) = ax.errorbar(freq[i], flux[i], flux_err[i], marker = 's',elinewidth=2,  markersize=10, color = 'DarkGoldenRod', linestyle='none', label = 'ATCA X',markeredgecolor='none')
                for cap in caps:
                    cap.set_color('DarkGoldenRod') 
                    cap.set_markeredgewidth(2)    
            else:
                ax.errorbar(freq[i], flux[i], flux_err[i], marker = 'o',elinewidth=2, markersize=10, color = 'MediumSeaGreen', linestyle='none', label = 'Data',markeredgecolor='none')
            # Elimanating doubled up legend values.
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(),loc='lower center', fontsize=15)


    else:   
        ax.errorbar(freq, flux, flux_err, marker = '.', color = 'darkgreen', linestyle='none', label = 'Data')
    
    if grid == True:
        ax.grid(which='both')
        if resid == True:
            ax1.grid(axis = 'x',which = 'both')

    # Make inset plot 

    # inset_ax = fig.add_axes([0.72,0.72,0.15,0.15])
    # xmin = -2
    # xmax = 2
    # ymin = -3
    # ymax = 2
    # inset_ax.set_xlim([xmin, xmax])
    # inset_ax.set_ylim([ymin, ymax])
    # inset_ax.yaxis.set_ticks([-3,-2.5,-2,-1.5,-1.,-0.5,0,0.5,1,1.5,2])

    # inset_ax.xaxis.set_major_formatter(ticker.NullFormatter())   # remove the major ticks
    # inset_ax.xaxis.set_minor_formatter(ticker.FuncFormatter(ticks_format_x))
    # inset_ax.yaxis.set_major_formatter(ticker.NullFormatter())   # remove the major ticks
    # inset_ax.yaxis.set_minor_formatter(ticker.FuncFormatter(ticks_format_y))
    # # inset_ax.axhline(linewidth=4)
    # # inset_ax.axvline(linewidth=4)

    # compx = np.linspace(xmin + 0.1*xmin,xmax +0.1*xmax)
    # compy = np.zeros(len(compx))
    # inset_ax.plot(compx,compy,linestyle = '-',color = 'k',linewidth = 2)
    # compy_2 = np.linspace(ymin + 0.1*ymin,ymax + 0.1*ymax)
    # compx_2 = np.zeros(len(compy))
    # inset_ax.plot(compx_2,compy_2,linestyle = '-',color = 'k',linewidth = 2)
    # y_diag = compx
    # inset_ax.plot(compx,y_diag,linestyle = '-',color = 'k',linewidth = 2)

    # # wedge
    # gps_x = np.arange(-0.5,0.11,0.01)
    # gps_line = 0.833*gps_x - 0.0833
    # # inset_ax.plot(gps_x,gps_line,linestyle = '-',color = 'k',linewidth = 1)
    # gps_y_vert = np.linspace(ymin + 0.1*ymin,-0.5)
    # gps_y_vert2 = np.linspace(ymin,2.1)
    # gps_x_vert = np.ones(len(gps_y_vert)) * -0.5
    # gps_x_vert2 = np.ones(len(gps_y_vert2)) * 0.1
    # # inset_ax.plot(gps_x_vert,gps_y_vert,linestyle = '-',color = 'k',linewidth = 1)
    # inset_ax.plot(gps_x_vert2,gps_y_vert2,linestyle = '-',color = 'k',linewidth = 1)

    # # GPS sel line

    # gps_x_sel = np.arange(-3.1,2.1,0.01)
    # gps_y_sel = np.ones(len(gps_x_sel))*-0.5
    # inset_ax.plot(gps_x_sel,gps_y_sel,linestyle = '-',color = 'k',linewidth = 1)

    # inset_ax.scatter(alpha_low, alpha_high,s=40, color='b',zorder=10)

    # inset_ax.tick_params(axis='both',which='both',bottom='off',top='off',left='off',right='off')

    if savefig == True:
        # Make a figures directory if there does not exist one and save the figures there.
        # if title == "No title provided.":
        #     for i in plt.get_fignums():
        #         if i == 0:
        #             print "Title names not provided. Graphs will be saved with figure numbers."
        #         title = 'Figure'+ str(plt.get_fignums()[-1])
        if not os.path.exists(os.getcwd()+'/seds'):
            os.makedirs(os.getcwd()+'/seds')
            print('Creating directory ', os.getcwd()+'/seds/ and saving figures in png format with title names.')
        plt.savefig('seds/'+str(name.replace(' ','_'))+'.png', bbox_inches='tight')
    plt.show()

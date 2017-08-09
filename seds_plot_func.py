# Joe Callingham 2017
# modified by MJ Rose 2017

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from collections import OrderedDict
import os
import re
from matplotlib import rc # To plot labels in serif rather than default.
import matplotlib.patches as mpatches
import mimic_alpha
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['serif'],'size':18})


from matplotlib.colors import colorConverter as cC
import numpy as np

def _to_rgb(c):
    """
    Convert color *c* to a numpy array of *RGB* handling exeption
    Parameters
    ----------
    c: Matplotlib color
        same as *color* in *colorAlpha_to_rgb*
    output
    ------
    rgbs: list of numpy array
        list of c converted to *RGB* array
    """
    if(isinstance(c,str)):  #if1: if c is a single element (number of string)
        rgbs = [np.array(cC.to_rgb(c)),]  #list with 1 RGB numpy array

    else:  #if1, else: if is more that one element

        try:   #try1: check if c is numberic or not
            np.array(c) + 1

        except (TypeError, ValueError):  #try1: if not numerics is not (only) RGB or RGBA colors
            #convert the list/tuble/array of colors into a list of numpy arrays of RGB
            rgbs = [np.array( cC.to_rgb(i)) for i in c]

        except Exception as e:  #try1: if any other exception raised
            print("Unexpected error: {}".format(e))
            raise e #raise it

        else:  #try1: if the colors are all numberics

            arrc = np.array(c)  #convert c to a numpy array
            arrcsh = arrc.shape  #shape of the array 

            if len(arrcsh)==1:  #if2: if 1D array given 
                if(arrcsh[0]==3 or arrcsh[0]==4):  #if3: if RGB or RBGA
                    rgbs = [np.array(cC.to_rgb(c)),]  #list with 1 RGB numpy array
                else:   #if3, else: the color cannot be RBG or RGBA
                    raise ValueError('Invalid rgb arg "{}"'.format(c))
                #end if3
            elif len(arrcsh)==2:  #if2, else: if 2D array
                if(arrcsh[1]==3 or arrcsh[1]==4):  #if4: if RGB or RBGA
                    rgbs = [np.array(cC.to_rgb(i)) for i in c]  #list with RGB numpy array
                else:   #if4, else: the color cannot be RBG or RGBA
                    raise ValueError('Invalid list or array of rgb')
                #end if4
            else:  #if2, else: if more dimention
                raise ValueError('The rgb or rgba values must be contained in a 1D or 2D list or array')
            #end if2
        #end try1
    #end if1

    return rgbs

def _is_number(s):
    """
    Check if *c* is a number (from
    http://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-in-python)
    Parameters
    ----------
    c: variable
    output
    ------
    true if c is a number
    false otherwise
    """
    try:
        float(s) # for int, long and float
    except ValueError:
        return False
    return True

def _check_alpha(alpha, n):
    """
    Check if alpha has one or n elements and if they are numberics and between 0 and 1
    Parameters
    ----------
    alpha: number or list/tuple/numpy array of numbers
        values to check
    output
    ------
    alpha: list of numbers 
        if all elements numberics and between 0 and 1
    """

    alpha = np.array(alpha).flatten()  #convert alpha to a flattened array
    if(alpha.size == 1):  #if1: alpha is one element
        if(_is_number(alpha) == False or alpha < 0 or alpha > 1):
            raise ValueError("'alpha' must be a float with value between 0 and 1, included") 
        else:
            alpha = [alpha for i in range(n)]  #replicate the alphas len(colors) times
    elif(alpha.size==n):  #if1, else: if alpha is composed of len(colors) elements
        try:  #check if all alphas are numbers
            alpha+1 
        except TypeError:
            raise ValueError("All elements of alpha must be a float with value between 0 and 1, included") 
        else:
            if((alpha < 0).any() or (alpha > 1).any()):
                raise ValueError("'alpha' must be a float with value between 0 and 1, included") 
    else:  #if1, else: if none of the previous cases
        raise ValueError("Alpha must have either one element or as many as 'colors'")
    #end if1
    return alpha

def colorAlpha_to_rgb(colors, alpha, bg='w'):
    """
    Given a Matplotlib color and a value of alpha, it returns 
    a RGB color which mimic the RGBA colors on the given background
    Parameters
    ----------
    colors: Matplotlib color (documentation from matplotlib.colors.colorConverter.to_rgb), 
        list/tuple/numpy array of colors
        Can be an *RGB* or *RGBA* sequence or a string in any of
        several forms:
        1) a letter from the set 'rgbcmykw'
        2) a hex color string, like '#00FFFF'
        3) a standard name, like 'aqua'
        4) a float, like '0.4', indicating gray on a 0-1 scale
        if *color* is *RGBA*, the *A* will simply be discarded.
    alpha: float [0,1] or list/tuple/numpy array with len(colors) elements
        Value of alpha to mimic. 
    bg: Matplotlib color (optional, default='w')
        Color of the background. Can be of any type shown in *color*
    output
    ------
    rgb: *RGB* color 
    example
    -------
    import mimic_alpha as ma
    print(ma.colorAlpha_to_rgb('r', 0.5))
    >>> [array([ 1. ,  0.5,  0.5])]
    print(ma.colorAlpha_to_rgb(['r', 'g'], 0.5)) 
    >>> [array([ 1. ,  0.5,  0.5]), array([ 0.5 ,  0.75,  0.5 ])]
    print(ma.colorAlpha_to_rgb(['r', 'g'], [0.5, 0.3])) 
    >>> [array([ 1. ,  0.5,  0.5]), array([ 0.7 ,  0.85,  0.7 ])]
    print(ma.colorAlpha_to_rgb(['r', [1,0,0]], 0.5)) 
    >>> [array([ 1. ,  0.5,  0.5]), array([ 1. ,  0.5,  0.5])]
    print( ma.colorAlpha_to_rgb([[0,1,1], [1,0,0]], 0.5) ) 
    >>> [array([ 0.5,  1. ,  1. ]), array([ 1. ,  0.5,  0.5])]
    print(ma.colorAlpha_to_rgb(np.array([[0,1,1], [1,0,0]]), 0.5)) 
    >>> [array([ 0.5,  1. ,  1. ]), array([ 1. ,  0.5,  0.5])]
    print(ma.colorAlpha_to_rgb(np.array([[0,1,1], [1,0,0]]), 0.5, bg='0.5')) 
    >>> [array([ 0.25,  0.75,  0.75]), array([ 0.75,  0.25,  0.25])]
    """
    colors = _to_rgb(colors)  #convert the color and save in a list of np arrays
    bg = _to_rgb(bg)[0]#np.array(cC.to_rgb(bg))   #convert the background
    
    #check if alpha has 1 or len(colors) elements and return a list of len(color) alpha 
    alpha = _check_alpha(alpha, len(colors))
    #interpolate between background and color 
    rgb = [(1.-a) * bg + a*c for c,a in zip(colors, alpha)][0]
    return rgb



cdict_models = {'singhomobremss':'red',
        'singinhomobremssbreakexp':'maroon',
        'singinhomobremssbreak': 'orangered',
        'singinhomobremss':'darkturquoise',
        'singinhomobremsscurve':'#4682b4',
        'doubhomobremss':'saddlebrown',
        'doubhomobremss':'Chocolate',
        'doubhomobremssbreak':'olive',
        'doubhomobremssbreak':'DarkGoldenRod',
        'singSSA':'orchid',
        'singSSAcurve':'darkmagenta',
        'singSSAbreak':'indigo',
        'doubSSA':'navy',
        'doubSSAbreakexp':'MediumSeaGreen',
        'doubSSAbreak':'black',
        'powlaw': 'DarkOrange',
        'powlawbreak':'dogerblue',
        'internalbremss':'sienna',
        'curve':'k'
            } 

# Defining plotting routine

def sed(models,model_labels,paras,freq,flux,flux_err, name,
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
        ax1.set_xlabel('Frequency (MHz)', fontsize = 35)
        ax1.set_ylabel(r'$\chi$', fontsize = 35)
        ax1.xaxis.labelpad = 15
        ax1.set_xlim(70., 21000.)
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
            #print ('Model is not in colour dictionary. Defaulting to dark orange.')
            color = 'DarkOrange'

        ax.plot(freq_cont, models[i](freq_cont, *paras[i]), color = color, linestyle='-', lw=2, label = model_labels[i])
        
        if resid == True:
            model_points = models[i](freq,*paras[i])
            residuals = flux-model_points
            chi_sing = residuals/flux_err
            chi_sing_err = np.ones(len(freq)) # Chi errors are 1.
            # ax1.errorbar(freq,residuals,flux_err,color = color, linestyle='none',marker = '.')
            ax1.errorbar(freq,chi_sing,chi_sing_err,color = color, linestyle='none',marker = '.')
            compx = np.linspace(70.-0.1*min(freq),max(freq)+0.1*max(freq))
            compy = np.zeros(len(compx))
            ax1.plot(compx,compy,linestyle = '--',color = 'gray',linewidth = 2)
            #ax1.set_xlim(min(freq_cont), max(freq_cont))
            # ax1.set_ylim(min(chi_sing)-0.2*min(chi_sing), max(chi_sing)+0.2*max(chi_sing))


        
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
                if value in [9.0, 60., 70, 90., 900.,700. ,200,400,600,700,800,2000,4000,6000,7000,8000,9000,10000]: # This will remove 90 and 900 MHz, replace number for anything you don't want to appear.
                    return ''
                if exp == 0 or exp == 1 or exp == 2 or exp ==3 or exp == 4:   
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
                if value in [9.0, 60., 70,90., 900.,700.,200,400,600,700,800,2000,4000,6000,7000,9000,10000]: # This will remove 90 and 900 MHz, replace number for anything you don't want to appear.
                    return ''
                if exp == 0 or exp == 1 or exp == 2 or exp ==3 or exp ==4 or exp==5:   
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
                if value in np.array([0.03,0.05,0.07,0.09,0.3,0.31,0.3,0.6,0.8,0.9,5,6,7,9,13,15,17,19,23,25,27,29]): # This will remove 90 and 900 MHz, replace number for anything you don't want to appear.
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
                if value in np.array([0.03,0.05,0.07,0.09,0.2,0.3,0.5,0.7,0.9,3,5,6,7,9,13,15,17,19,23,25,27,29]): # This will remove 90 and 900 MHz, replace number for anything you don't want to appear.
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
            elif freq[i] in [4850,8640]:
                (_, caps, _) = ax.errorbar(freq[i], flux[i], flux_err[i], marker = 'd',elinewidth=2, markersize=10,  color = 'darkgreen', linestyle='none', label = 'ATPMN',markeredgecolor='none')
                for cap in caps:
                    cap.set_color('darkgreen') 
                    cap.set_markeredgewidth(2)

            elif freq[i] in [4800.001,8641.001, 19904.001]:
                (_, caps, _) = ax.errorbar(freq[i], flux[i], flux_err[i], marker = 'd',elinewidth=2, markersize=10,  color = 'MediumSeaGreen', linestyle='none', label = 'AT20G',markeredgecolor='none')
                for cap in caps:
                    cap.set_color('MediumSeaGreen') 
                    cap.set_markeredgewidth(2)
    
            # elif 1290 <= freq[i] <= 3030:
            #     (_, caps, _) = ax.errorbar(freq[i], flux[i], flux_err[i], marker = 's',elinewidth=2, markersize=8,  color = 'darkmagenta', linestyle='none', label = 'ATCA L',markeredgecolor='none')
            #     for cap in caps:
            #         cap.set_color('darkmagenta') 
            #         cap.set_markeredgewidth(2)            
            # elif 4500 <= freq[i] <= 6430:
            #     (_, caps, _) = ax.errorbar(freq[i], flux[i], flux_err[i], marker = '^',elinewidth=2, markersize=10,  color = 'indigo', linestyle='none', label = 'ATCA C',markeredgecolor='none')
            #     for cap in caps:
            #         cap.set_color('indigo') 
            #         cap.set_markeredgewidth(2)
            # elif 8070 <= freq[i] <= 9930:
            #     (_, caps, _) = ax.errorbar(freq[i], flux[i], flux_err[i], marker = 's',elinewidth=2,  markersize=10, color = 'DarkGoldenRod', linestyle='none', label = 'ATCA X',markeredgecolor='none')
            #     for cap in caps:
            #         cap.set_color('DarkGoldenRod') 
            #         cap.set_markeredgewidth(2)    
            # else:
            #     ax.errorbar(freq[i], flux[i], flux_err[i], marker = 'o',elinewidth=2, markersize=10, color = 'MediumSeaGreen', linestyle='none', label = 'Data',markeredgecolor='none')
            else:
                try:
                    ind = np.where((freq >= 1290.) & (freq <= 3030.))
                    flux_fill_upper = flux[ind] + flux_err[ind]
                    flux_fill_lower = flux[ind] - flux_err[ind]
                    ax.fill_between(freq[ind], flux_fill_lower, flux_fill_upper, facecolor = colorAlpha_to_rgb('dodgerblue',0.5), alpha=0.3, edgecolor='none', label = 'ATCA L')
                    ax.plot([], [], color=colorAlpha_to_rgb('dodgerblue',0.5), linewidth=10,label = 'ATCA L')

                    ind = np.where((freq >= 4500.) & (freq <= 6430.))
                    flux_fill_upper = flux[ind] + flux_err[ind]
                    flux_fill_lower = flux[ind] - flux_err[ind]
                    ax.fill_between(freq[ind], flux_fill_lower, flux_fill_upper, facecolor = colorAlpha_to_rgb('indigo',0.5), alpha=0.3, edgecolor='none', label = 'ATCA C')
                    ax.plot([], [], color=colorAlpha_to_rgb('indigo',0.5), linewidth=10,label = 'ATCA C')

                    ind = np.where((freq >= 8070.) & (freq <= 9930.))
                    flux_fill_upper = flux[ind] + flux_err[ind]
                    flux_fill_lower = flux[ind] - flux_err[ind]
                    ax.fill_between(freq[ind], flux_fill_lower, flux_fill_upper, facecolor = colorAlpha_to_rgb('DarkGoldenRod',0.5), edgecolor='none', label = 'ATCA X')
                    ax.plot([], [], color=colorAlpha_to_rgb('DarkGoldenRod',0.5), linewidth=10,label = 'ATCA X')
                except:
                    pass




            # Elimanating doubled up legend values.
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            bbox = (1.2,0.7)
            ax.legend(by_label.values(), by_label.keys(),bbox_to_anchor=bbox, fontsize=15)
            


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
    return(fig)


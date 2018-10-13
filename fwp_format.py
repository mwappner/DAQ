# -*- coding: utf-8 -*-
"""
This module formats plots and values for Latex.

Some of its most useful tools are:

value_error : function
    Rounds up value and error of a measure. Also makes a latex string.
plot_style : function
    Gives a specific style to figure.

@author: Vall
"""

from matplotlib import rcParams, ticker
import matplotlib.pyplot as plt
from tkinter import Tk

#%%

def copy(string):
    """Copies a string to the clipboard.
    
    Parameters
    ----------
    string : str
        The string to be copied.
    
    Returns
    -------
    nothing
    
    """
    
    r = Tk()
    r.withdraw()
    r.clipboard_clear()
    r.clipboard_append(string)
    r.update() # now it stays on the clipboard
    r.destroy()
    
    print("Copied")

#%%

def error_value(X, dX, error_digits=2, units='',
                string_scale=True, one_point_scale=False, legend=False):
    """Rounds up value and error of a measure. Also makes a latex string.
    
    This function takes a measure and its error as input. Then, it 
    rounds up both of them in order to share the same amount of decimal 
    places.
    
    After that, it generates a latex string containing the rounded up 
    measure. For that, it can rewrite both value and error so that the 
    classical prefix scale of units can be applied.
    
    Parameters
    ----------
    X : float
        Measurement's value.
    dX : float
        Measurement's associated error.
    error_digits=2 : int, optional.
        Desired number of error digits.
    units='' : str, optional.
        Measurement's units.
    string_scale=True : bool, optional.
        Whether to apply the classical prefix scale or not.        
    one_point_scale=False : bool, optional.
        Applies prefix with one order less.
    legend=False : bool, optional.
        Says whether it is for the legend of a plot or not.
    
    Returns
    -------
    latex_str : str
        Latex string containing value and error.
    
    Examples
    --------
    >> error_value(1.325412, 0.2343413)
    '(1.33$\\pm$0.23)'
    >> error_value(1.325412, 0.2343413, error_digits=3)
    '(1.325$\\pm$0.234)'
    >> error_value(.133432, .00332, units='V')
    '\\mbox{(133.4$\\pm$3.3) mV}'
    >> error_value(.133432, .00332, one_point_scale=True, units='V')
    '\\mbox{(0.1334$\\pm$0.0033) V}'
    >> error_value(.133432, .00332, string_scale=False, units='V')
    '\\mbox{(1.334$\\pm$0.033)$10^-1$ V}'
    
    See Also
    --------
    copy
    
    """
    
    # First, I string-format the error to scientific notation with a 
    # certain number of digits
    if error_digits >= 1:
        aux = '{:.' + str(error_digits) + 'E}'
    else:
        print("Unvalid 'number_of_digits'! Changed to 1 digit")
        aux = '{:.0E}'
    error = aux.format(dX)
    error = error.split("E") # full error (string)
    
    error_order = int(error[1]) # error's order (int)
    error_value = error[0] # error's value (string)

    # Now I string-format the measure to scientific notation
    measure = '{:E}'.format(X)
    measure = measure.split("E") # full measure (string)
    measure_order = int(measure[1]) # measure's order (int)
    measure_value = float(measure[0]) # measure's value (string)
    
    # Second, I choose the scale I will put both measure and error on
    # If I want to use the string scale...
    if -12 <= measure_order < 12 and string_scale:
        prefix = ['p', 'n', r'$\mu$', 'm', '', 'k', 'M', 'G']
        scale = [-12, -9, -6, -3, 0, 3, 6, 9, 12]
        for i in range(8):
            if not one_point_scale:
                if scale[i] <= measure_order < scale[i+1]:
                    prefix = prefix[i] # prefix to the unit
                    scale = scale[i] # order of both measure and error
                    break
            else:
                if scale[i]-1 <= measure_order < scale[i+1]-1:
                    prefix = prefix [i]
                    scale = scale[i]
                    break
        used_string_scale = True
    # ...else, if I don't or can't...
    else:
        scale = measure_order
        prefix = ''
        used_string_scale = False
    
    # Third, I actually scale measure and error according to 'scale'
    # If error_order is smaller than scale...
    if error_order < scale:
        if error_digits - error_order + scale - 1 >= 0:
            aux = '{:.' + str(error_digits - error_order + scale - 1)
            aux = aux + 'f}'
            error_value = aux.format(
                    float(error_value) * 10**(error_order - scale))
            measure_value = aux.format(
                    measure_value * 10**(measure_order - scale))
        else:
            error_value = float(error_value) * 10**(error_order - scale)
            measure_value = float(measure_value)
            measure_value = measure_value * 10**(measure_order - scale)
    # Else, if error_order is equal or bigger than scale...
    else:
        aux = '{:.' + str(error_digits - 1) + 'f}'
        error_value = aux.format(
                float(error_value) * 10**(error_order - scale))
        measure_value = aux.format(
                float(measure_value) * 10**(measure_order - scale))
    
    # Forth, I generate a latex string. Ex.: '(1.34$pm$0.32) kV'
    latex_str = r'({}$\pm${})'.format(measure_value, error_value)
    if not used_string_scale and measure_order != 0:
        latex_str = latex_str + r'$10^{:.0f}$'.format(scale)      
    elif used_string_scale:
        latex_str = latex_str + ' ' + prefix
    if units != '':
        if latex_str[-1] == ' ':
            latex_str = latex_str + units
        else:
            latex_str = latex_str + ' ' + units
    if units != '' or prefix:
        if not legend:
            latex_str = r'\mbox{' + latex_str + '}'
                       
    return latex_str

#%%

def plot_style(figure_id=None, **kwargs):
    """Gives a specific style to figure.
    
    This function...
        ...increases font size;
        ...increases linewidth;
        ...increases markersize;
        ...gives format to axis ticks if specified;
        ...stablishes new figure dimensions if specified;
        ...activates grid.
    
    Parameters
    ----------
    figure_id : int, optional.
        ID of the figure where the text will be printed.
        If none is given, the current figure is taken as default.
    
    Other Parameters
    ----------------
    xaxisformat : format-like str, optional.
        Used to update x axis ticks format; i.e.: '%.2e'
    yaxisformat : format-like str, optional.
        Used to update y axis ticks format; i.e.: '%.2e'
    dimensions: list with length 4, optional.
        Used to update plot dimensions: [xmin, xmax, ymin, ymax]. Each 
        one should be a number expressed as a fraction of current 
        dimensions.
    
    See Also
    --------
    matplotlib.pyplot.axis
    matplotlib.pyplot.gcf
    
    """
    
    if figure_id is None:
        fig = plt.gcf()
    else:
        fig = plt.figure(figure_id)
    ax = fig.axes
    
    rcParams.update({'font.size': 14})
    rcParams.update({'lines.linewidth': 3})
    rcParams.update({'lines.markersize': 6})
    
    kwargs_list = ['xaxisformat', 'yaxisformat', 'dimensions']
    for key in kwargs_list:
        try:
            kwargs[key]
        except KeyError:
            kwargs[key] = None
    
    if kwargs['xaxisformat'] is not None:
        for a in ax:
            a.xaxis.set_major_formatter(ticker.FormatStrFormatter(
                kwargs['xaxisformat']))
        
    if kwargs['yaxisformat'] is not None:
        for a in ax:
            a.yaxis.set_major_formatter(ticker.FormatStrFormatter(
                kwargs['yaxisformat']))
    
    if kwargs['dimensions'] is not None:
        for a in ax:
            box = a.get_position()
            a.set_position([kwargs['dimensions'][0]*box.x0,
                            kwargs['dimensions'][1]*box.y0,
                            kwargs['dimensions'][2]*box.width,
                            kwargs['dimensions'][3]*box.height])
    
    for a in ax:
        a.grid()
    
    plt.show()

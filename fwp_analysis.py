# -*- coding: utf-8 -*-
"""
This module contains tools for data analysis.

Some of its most useful tools are:
    
mean : function
    Returns average or weighted average and standard deviation.
linear_fit : function
    Applies linear fit and returns m, b and Rsq. Can also plot it.
nonlinear_fit : function
    Applies nonlinear fit and returns parameters and Rsq. Plots it.

@author: Vall
"""

import fwp_format as fmt
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
from scipy.optimize import curve_fit

#%%

def rms(data):
    """Takes a list or array and returns RMS value.
    
    Parameters
    ---------
    data : array or list
        Data to be analized.
    
    Returns
    -------
    float
        RMS value of analized data.
    
    """
    
    import numpy as np
    
    return np.sqrt(np.mean((np.array(data))**2))

#%%

def mean(X, dX=None):
    """Returns average or weighted average and standard deviation.
    
    If dX is given, it returns weighted average (weight=1/dX**2) of 
    data. If not, it returns common average.

    Parameters
    ----------
    X : list, np.array
        Data.
    dX=None : list, np.array
        Data's error.
    
    Returns
    -------
    (mean, std): tuple
        This tuple contains data's mean and standard deviation. If dX is 
        given, it returns weighted values (weight=1/dX**2).
    
    Raises
    ------
    "The X data is not np.array-like" : TypeError
        If X can't be easily converted to numpy array.
    "The dX data is not np.array-like" : TypeError
        If dX can't be easily converted to numpy array.
    "The dX array's length should match X's" : IndexError
        If dX's length doesn't match X's.
    
    """
    
    
    if not isinstance(X, np.ndarray):
        try:
            X = np.array(X)
        except:
            raise TypeError("The X data is not np.array like")

    if dX is not None:
        if not isinstance(dX, np.ndarray):
            try:
                dX = np.array(dX)
            except:
                raise TypeError("The dX data is not np.array like")
        if len(dX) != len(X):
            raise IndexError("The dX array's length should match X's")

        mean = np.average(X, weights=1/dX**2)
        variance = np.average((X-mean)**2, weights=1/dX**2)
        return (mean, sqrt(variance))
    
    else:
        return (np.mean(X), np.std(X))

#%%

def linear_fit(X, Y, dY=None, showplot=True,
               plot_some_errors=(False, 20), **kwargs):
    """Applies linear fit and returns m, b and Rsq. Can also plot it.
    
    By default, it applies minimum-square linear fit 'y = m*x + b'. If 
    dY is specified, it applies weighted minimum-square linear fit.
    
    Parameters
    ----------
    X : np.array, list
        Independent X data to fit.
    Y : np-array, list
        Dependent Y data to fit.
    dY : np-array, list
        Dependent Y data's associated error.
    shoplot : bool
        Says whether to plot or not.
    plot_some_errors : tuple (bool, int)
        Says wehther to plot only some error bars (bool) and specifies 
        the number of error bars to plot.
    
    Returns
    -------
    rsq : float
        Linear fit's R Square Coefficient.
    (m, dm): tuple (float, float)
        Linear fit's slope: value and associated error, both as float.
    (b, db): tuple (float, float)
        Linear fit's origin: value and associated error, both as float.
        
    Other Parameters
    ----------------
    txt_position : tuple (horizontal, vertical), optional
        Indicates the parameters' text position. Each of its values 
        should be a number (distance in points measured on figure). 
        But vertical value can also be 'up' or 'down'.
    mb_units : tuple (m_units, b_units), optional
        Indicates the parameter's units. Each of its values should be a 
        string.
    mb_error_digits : tuple (m_error_digits, b_error_digits), optional
        Indicates the number of digits to print in the parameters' 
        associated error. Default is 3 for slope 'm' and 2 for intercept 
        'b'.
    mb_string_scale : tuple (m_string_scale, b_string_scale), optional
        Indicates whether to apply string prefix's scale to printed 
        parameters. Each of its values should be a bool; i.e.: 'True' 
        means 'm=1230.03 V' with 'dm = 10.32 V' would be printed as 
        'm = (1.230 + 0.010) V'. Default is '(False, False)'.
    rsq_decimal_digits : int, optional.
        Indicates the number of digits to print in the Rsq. Default: 3.
        
    Warnings
    --------
    The returned Rsq doesn't take dY weights into account.
    
    """

    # ¿Cómo hago Rsq si tengo pesos?
    
    if dY is None:
        W = None
    else:
        W = 1/dY**2
        
    fit_data = np.polyfit(X,Y,1,cov=True,w=W)
    
    m = fit_data[0][0]
    dm = sqrt(fit_data[1][0,0])
    b = fit_data[0][1]
    db = sqrt(fit_data[1][1,1])
    rsq = 1 - sum( (Y - m*X - b)**2 )/sum( (Y - np.mean(Y))**2 )

    try:
        kwargs['text_position']
    except KeyError:
        if m > 1:
            aux = 'up'
        else:
            aux = 'down'
        kwargs['text_position'] = (.02, aux)

    if showplot:

        plt.figure()
        if dY is None:
            plt.plot(X, Y, 'b.', zorder=0)
        else:
            if plot_some_errors[0] == False:
                plt.errorbar(X, Y, yerr=dY, linestyle='b', marker='.',
                             ecolor='b', elinewidth=1.5, zorder=0)
            else:
                plt.errorbar(X, Y, yerr=dY, linestyle='-', marker='.',
                             color='b', ecolor='b', elinewidth=1.5,
                             errorevery=len(Y)/plot_some_errors[1], 
                             zorder=0)
        plt.plot(X, m*X+b, 'r-', zorder=100)
        plt.legend(["Ajuste lineal ponderado","Datos"])
        
        kwargs_list = ['mb_units', 'mb_string_scale', 
                       'mb_error_digits', 'rsq_decimal_digits']
        kwargs_default = [('', ''), (False, False), (3, 2), 3]
        for key, value in zip(kwargs_list, kwargs_default):
            try:
                kwargs[key]
            except KeyError:
                kwargs[key] = value
        
        if kwargs['text_position'][1] == 'up':
            vertical = [.9, .82, .74]
        elif kwargs['text_position'][1] == 'down':
            vertical = [.05, .13, .21]
        else:
            if kwargs['text_position'][1] <= .08:
                fact = .08
            else:
                fact = -.08
            vertical = [kwargs['text_position'][1]+fact*i for i in range(3)]
        
        plt.annotate('m = {}'.format(fmt.error_value(
                        m, 
                        dm,
                        error_digits=kwargs['mb_error_digits'][0],
                        units=kwargs['mb_units'][0],
                        string_scale=kwargs['mb_string_scale'][0],
                        one_point_scale=True,
                        legend=True)),
                    (kwargs['text_position'][0], vertical[0]),
                    xycoords='axes fraction')
        plt.annotate('b = {}'.format(fmt.error_value(
                        b, 
                        db,
                        error_digits=kwargs['mb_error_digits'][1],
                        units=kwargs['mb_units'][1],
                        string_scale=kwargs['mb_string_scale'][1],
                        one_point_scale=True,
                        legend=True)),
                    (kwargs['text_position'][0], vertical[1]),
                    xycoords='axes fraction')
        rsqft = r'$R^2$ = {:.' + str(kwargs['rsq_decimal_digits']) + 'f}'
        plt.annotate(rsqft.format(rsq),
                    (kwargs['text_position'][0], vertical[2]),
                    xycoords='axes fraction')
        
        plt.show()

    return rsq, (m, dm), (b, db)

#%%

def nonlinear_fit(X, Y, fitfunction, initial_guess=None, dY=None, 
                  showplot=True, plot_some_errors=(False, 20), 
                  **kwargs):
    """Applies nonlinear fit and returns parameters and Rsq. Plots it.
    
    By default, it applies minimum-square fit. If dY is specified, it 
    applies weighted minimum-square fit.
    
    Parameters
    ----------
    X : np.array, list
        Independent X data to fit.
    Y : np-array, list
        Dependent Y data to fit.
    fitfunction : function
        The function you want to apply. Its arguments must be 'X' as 
        np.array followed by the other parameters 'a0', 'a1', etc as 
        float. Must return only 'Y' as np.array.
    initial_guess=None : list, optional
        A list containing a initial guess for each parameter.
    dY : np-array, list, optional
        Dependent Y data's associated error.
    shoplot : bool
        Says whether to plot or not.
    plot_some_errors : tuple (bool, int)
        Says wehther to plot only some error bars (bool) and specifies 
        the number of error bars to plot.
    
    Returns
    -------
    rsq : float
        Fit's R Square Coefficient.
    parameters : list of tuples
        Fit's parameters, each as a tuple containing value and error, 
        both as tuples.
    
    Other Parameters
    ----------------
    txt_position : tuple (horizontal, vertical), optional
        Indicates the parameters' text position. Each of its values 
        should be a number (distance in points measured on figure). 
        But vertical value can also be 'up' or 'down'.
    par_units : list, optional
        Indicates the parameters' units. Each of its values should be a 
        string.
    par_error_digits : list, optional
        Indicates the number of digits to print in the parameters' 
        associated error. Default is 3 for all of them.
    par_string_scale : list, optional
        Indicates whether to apply string prefix's scale to printed 
        parameters. Each of its values should be a bool. Default is 
        False for all of them.
    rsq_decimal_digits : int, optional.
        Indicates the number of digits to print in the Rsq. Default: 3.
        
    Warnings
    --------
    The returned Rsq doesn't take dY weights into account.
    
    """
    
    if not isinstance(X, np.ndarray):
        raise TypeError("X should be a np.array")
    if not isinstance(Y, np.ndarray):
        raise TypeError("Y should be a np.array")
    if not isinstance(dY, np.ndarray) and dY is not None:
        raise TypeError("dY shouuld be a np.array")
    if len(X) != len(Y):
        raise IndexError("X and Y must have same lenght")
    if dY is not None and len(dY) != len(Y):
        raise IndexError("dY and Y must have same lenght")
    
    if dY is None:
        W = None
    else:
        W = 1/dY**2
    
    parameters, covariance = curve_fit(fitfunction, X, Y,
                                       p0=initial_guess, sigma=W)  
    rsq = sum( (Y - fitfunction(X, *parameters))**2 )
    rsq = rsq/sum( (Y - np.mean(Y))**2 )
    rsq = 1 - rsq

    if showplot:
        
        plt.figure()
        if dY is None:
            plt.plot(X, Y, 'b.', zorder=0)
        else:
            if plot_some_errors[0] == False:
                plt.errorbar(X, Y, yerr=dY, linestyle='b', marker='.',
                             ecolor='b', elinewidth=1.5, zorder=0)
            else:
                plt.errorbar(X, Y, yerr=dY, linestyle='-', marker='.',
                             color='b', ecolor='b', elinewidth=1.5,
                             errorevery=len(Y)/plot_some_errors[1], 
                             zorder=0)
        plt.plot(X, fitfunction(X, *parameters), 'r-', zorder=100)        
        plt.legend(["Ajuste lineal ponderado","Datos"])
        
        n = len(parameters)
        kwargs_list = ['text_position', 'par_units', 'par_string_scale', 
                       'par_error_digits', 'rsq_decimal_digits']
        kwargs_default = [(.02,'up'), ['' for i in range(n)], 
                          [False for i in range(n)], 
                          [3 for i in range(n)], 3]
        for key, value in zip(kwargs_list, kwargs_default):
            try:
                kwargs[key]
                if key != 'text_position':
                    try:
                        if len(kwargs[key]) != n:
                            print("Wrong number of parameters",
                                  "on '{}'".format(key))
                            kwargs[key] = value
                    except TypeError:
                        kwargs[key] = [kwargs[key] for i in len(n)]
            except KeyError:
                kwargs[key] = value
        
        if kwargs['text_position'][1] == 'up':
            vertical = [.9-i*.08 for i in range(n+1)]
        elif kwargs['text_position'][1] == 'down':
            vertical = [.05+i*.08 for i in range(n+1)]
        else:
            if kwargs['text_position'][1] <= .08:
                fact = .08
            else:
                fact = -.08
            vertical = [
                kwargs['text_position'][1]+fact*i for i in range(n+1)]
        
        for i in range(n):
            plt.annotate(
                    'a{} = {}'.format(
                        i,
                        fmt.error_value(
                            parameters[i], 
                            sqrt(covariance[i,i]),
                            error_digits=kwargs['par_error_digits'][i],
                            units=kwargs['par_units'][i],
                            string_scale=kwargs['par_string_scale'][i],
                            one_point_scale=True,
                            legend=True)),
                    (kwargs['text_position'][0], vertical[i]),
                    xycoords='axes fraction')
        rsqft = r'$R^2$ = {:.' + str(kwargs['rsq_decimal_digits'])+'f}'
        plt.annotate(rsqft.format(rsq),
                    (kwargs['text_position'][0], vertical[-i]),
                    xycoords='axes fraction')
        
        plt.show()
    
    parameters_error = np.array(
            [sqrt(covariance[i,i]) for i in range(n)])
    parameters = list(zip(parameters, parameters_error))
    
    return rsq, parameters
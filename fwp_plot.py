# -*- coding: utf-8 -*-
"""
This module contains tools for plotting.

@author: Vall
"""

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

#%%

def add_text(text, text_position='up', figure_id=None):
    """Prints some text on a matplotlib.pyplot figure.
    
    Parameters
    ----------
    text : str
        Text to be printed.
    text_position : tuple, str {'up', 'dowm'}
        Position of the text to be printed.
    figure_id=None : int
        ID of the figure where the text will be printed.
        If none is given, the current figure is taken as default.
    
    Returns
    -------
    nothing
    
    Yields
    ------
    matplotlib.annotation
    
    See Also
    --------
    plot_style
    matplotlib.pyplot.gcf
    
    """

    if figure_id is None:
        plt.gcf()
    else:
        plt.figure(figure_id)
    
    if text_position == 'up':
        plt.annotate(text, (0.02,0.9), xycoords="axes fraction")
    elif text_position == 'down':
        plt.annotate(text, (0.02,0.05), xycoords="axes fraction")
    else:
        plt.annotate(text, text_position, xycords="axes fraction")
    
    plt.show()

#%% More than one 2D plot

def graphs_2D(X, Y, lcolor='blue'):
    """Plots several lines with a given color scale.
    
    Parameters
    ----------
    X : np.array
        Data's X values. Can be a 1D n-array or a 2D (nxN) array, where 
        each column corresponds to a different data series.
    Y : np.array
        Data's Y values. Must be a 2D (nxN) array, where each column is 
        a series of data.
    lcolor : str {'blue', 'green', 'red', 'violet', 'mixt'}, optional.
        Plot lines' color scale.
    
    Returns
    -------
    nothing
    
    Yields
    ------
    matplotlib.figure
    
    Raises
    ------
    "Y should be a 2D array" : TypeError
        If Y doesn't have more than 1 column.
    "X, Y should have the same number of columns" : Type Error
        If X is not 1D and X, Y don't have the same number of columns.
    "X, Y should have the same number of rows" : Type Error
        If X, Y don't have the same number of rows.
    "leg should be..." : TypeError
        If leg isn't either a format string nor a list which length is 
        Y's number of columns.
    
    """
    
    try:
        Y_rows = len(Y[:,0])
        Y_cols = len(Y[0,:])
    except:
        raise TypeError("Y should be a 2D array")
    
    try:
        X_rows = len(X[:,0])
        X_cols = len(X[0,:])
    except:
        X_rows = len(X)
        X_cols = 1
        
    if X_cols != 1 and X_cols != Y_cols:
        raise TypeError("X, Y should have the same number of columns")
    
    if X_rows != Y_rows:
        raise TypeError("X, Y should have the same number of rows")
    
    if lcolor == 'blue':
        lcolor = [[0, 0, (Y_cols-i)/Y_cols] for i in range(Y_cols)]
    if lcolor == 'green':
        lcolor = [[0, (Y_cols-i)/Y_cols, 0] for i in range(Y_cols)]
    if lcolor == 'red':
        lcolor = [[(Y_cols-i)/Y_cols, 0, 0] for i in range(Y_cols)]
    if lcolor == 'violet':
        lcolor = [[(Y_cols-i)/Y_cols, 0, (Y_cols-i)/Y_cols] 
                    for i in range(Y_cols)]
    if lcolor == 'mixt':
        lcolor = [[(Y_cols-i)/Y_cols, 0, (i+1)/Y_cols] 
                    for i in range(Y_cols)]
    elif len(lcolor) != Y_cols:
        message = "leg should be a {}-array like".format(Y_cols)
        message = message + "or should be in {}".format(['blue',
                                                         'green',
                                                         'red',
                                                         'violet',
                                                         'mixt'])
        raise TypeError(message)
        
    plt.figure()
    if X_cols == 1:
        for i in range(Y_cols):
            plt.plot(X, Y[:,i], color=lcolor[i])
    else:
        for i in range(Y_cols):
            plt.plot(X[:,i], Y[:,i], color=lcolor[i])
    plt.show()

#%%

def animation_2D(X, Y, figure_id=None, frames_number=30,
                 label_function=lambda i : '{:.0f}'.format(i)):
    """Makes a series of plots into an animation.
    
    Parameters
    ----------
    X : np.array
        Data's X values. Can be a 1D n-array or a 2D (nxN) array, where 
        each column corresponds to a different data series.
    Y : np.array
        Data's Y values. Must be a 2D (nxN) array, where each column is 
        a series of data.
    frames_number=30 : int, optional
        Animation's number of frames.
    label_function : function, optional.
        Function that assigns frames' labels. Must take in one int 
        parameter and return one string.
    
    Returns
    -------
    animation : matplotlib.animation object
        Animation.
    
    See Also
    --------
    matplotlib.animation
    fwp_save.saveanimation
    
    """
    
    try:
        Y_rows = len(Y[:,0])
        Y_cols = len(Y[0,:])
    except:
        raise TypeError("Y should be a 2D array")
    
    try:
        X_rows = len(X[:,0])
        X_cols = len(X[0,:])
    except:
        X_rows = len(X)
        X_cols = 1
        
    if X_cols != 1 and X_cols != Y_cols:
        raise TypeError("X, Y should have the same number of columns")
    
    if X_rows != Y_rows:
        raise TypeError("X, Y should have the same number of rows")
    
    fig = plt.figure()
    ax = plt.axes(xlim=(np.amin(X),np.amax(X)), 
                  ylim=(np.amin(Y),np.amax(Y)))
    line, = ax.plot([], [])
    label = ax.text(0.02, 0.90, '', transform=ax.transAxes)
   
    def init():
        line.set_data([], [])
        label.set_text('')
        return line, label
    
    if X_cols == 1:
        def animate(i):
            line.set_data(X, Y[:,i])
            label.set_text(label_function(i))
            return line, label
    else:
        def animate(i):
            line.set_data(X[:,i], Y[:,i])
            label.set_text(label_function(i))
            return line, label        
    
    anim = animation.FuncAnimation(fig, 
                                   animate, 
                                   init_func=init, 
                                   frames=frames_number, 
                                   interval=frames_number*3, 
                                   blit=True)
    
    return anim;
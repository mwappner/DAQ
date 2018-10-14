# -*- coding: utf-8 -*-
"""
The 'fwp_plot' module contains tools for plotting.

Some of its most useful tools are:

animation_2D : function
	Makes a series of 2D plots into an animation.
animation_3D : function
	Makes a series of 3D plots into an animation.
add_style : function
	Gives a specific style to figure.

@author: Vall
"""

from matplotlib import rcParams, ticker, animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import winter, summer, spring, autumn, cool, hot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

#%%

def add_labels_3D(title=None, xlabel=None, ylabel=None, zlabel=None, 
                  figure_id=None, new_figure=False):
    """Labels a 3D graph's axis.
    
    Parameters
    ----------
    title=None : str, optional
        Plot's title.
    xlabel=None : str, optional
        Plot's X label.
    ylabel=None : str, optional
        Plot's Y label.    
    zlabel=None : str, optional
        Plot's Z label.
    figure_id=None : int, optional
        A matplotlib figure's ID. If none is specified, it uses the 
        current active figure.
    new_figure=False : bool, optional
        Indicates whether to make a new figure or not when 
        figure_id=None.
    
    Returns
    -------
    nothing
    
    Yields
    ------
    axis labels
    
    """
    
    if figure_id is not None:
        fig = plt.figure(figure_id)
    elif new_figure:
        fig = plt.figure()
    else:
        fig = plt.gcf()
    
    try:
        ax = fig.axes
        ax[0]
    except IndexError:
        ax = [plt.axes()]
    
    if xlabel is not None:
        for a in ax:
            a.set_xlabel(xlabel)
    if ylabel is not None:
        for a in ax:
            ax.set_ylabel(ylabel)
    if zlabel is not None:
        for a in ax:
            ax.set_zlabel(zlabel)

#%%

def add_style(figure_id=None, new_figure=False, **kwargs):
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
    figure_id : int, optional
        ID of the figure where the text will be printed.
        If none is given, the current figure is taken as default.
    new_figure=False : bool, optional
        Indicates whether to make a new figure or not when 
        figure_id=None.
    
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
    
    if figure_id is not None:
        fig = plt.figure(figure_id)
    elif new_figure:
        fig = plt.figure()
    else:
        fig = plt.gcf()

    try:
        ax = fig.axes
        ax[0]
    except IndexError:
        ax = [plt.axes()]
    
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
    "X, Y should have the same number of columns" : IndexError
        If X is not 1D and X, Y don't have the same number of columns.
    "X, Y should have the same number of rows" : IndexError
        If X, Y don't have the same number of rows.
    "lcolor should be..." : TypeError
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
        raise IndexError("X, Y should have the same number of columns")
    
    if X_rows != Y_rows:
        raise IndexError("X, Y should have the same number of rows")
    
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
        message = "lcolor should be a {}-array like".format(Y_cols)
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

def graph_3D(X, Y, Z, color_map='winter'):
    """Makes a 3D plot.

    Parameters
    ----------
    X : np.array
        Data's X coordinates. Should be a 1D array of size X_len.
    Y : np.array
        Data's Y coordinates. Should be a 1D array of size Y_len.
    Z : np.array
        Data on a 2D array of size (Y_len, X_len).
    color_map='winter' : str {'winter', 'summer', ...}
        Color map specifier. Should be on ['spring', 'autumn', 'cool', 
        'hot', 'winter', 'summer']. If not, must be imported from 
        matplotlib.cm
    
    Returns
    -------
    nothing
    
    Yields
    ------
    matplotlib.figure
    
    Raises
    ------
    "X should be array-like" : TypeError
        If len(X) raises error.
    "Y should be array-like" : TypeError
        If len(X) raises error.
    "X, Y should have the same legth" : IndexError
        If X, Y don't have the same length.
    "Z should be a 2D array" : TypeError
        If Z is not a 2D array.
    "color_map should be..." : TypeError
        If color_map is not an allowed string or matplotlib colormap.
        
    """
    
    try:
        X_len = len(X)
    except:
        raise TypeError("X should be array-like")

    try:
        Y_len = len(Y)
    except:
        raise TypeError("Y should be array-like")
        
    try:
        Z[:,0]
    except:
        raise TypeError("Z should be a 2D array")
    
    if X_len != Y_len:
        raise IndexError("X, Y should have the same length")
    
    key = ['winter', 'summer', 'autumn', 'spring', 'cool', 'hot']
    if color_map not in key:
        if not isinstance(color_map,
                          LinearSegmentedColormap):
            raise TypeError("color_map should be a matplotlib colormap",
                            " or should be on {}".format(key))
    
    X, Y = np.meshgrid(X, Y)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, cmap = color_map) #rstride=1, cstride=1
    #ax.contourf(X, Y, Z, cmap=winter)#zdir='z', offset=-2, cmap=winter)

#%%

def graphs_3D(X, Y, Z, Z2, X2=None, Y2=None, 
              color_map=['winter','summer']):
    """Makes two 3D plots on different subplots.

    Parameters
    ----------
    X : np.array
        Data's X coordinates. Should be a 1D array of size X_len.
    Y : np.array
        Data's Y coordinates. Should be a 1D array of size Y_len.
    Z : np.array
        Data on a 2D array of size (Y_len, X_len).
    Z2 : np.array
        2nd data on a 2D array.
    X2 : np.array, optional
        Data's 2nd X coordinates. Should be a 1D array.
    Y2 : np.array, optional
        Data's 2nd Y coordinates. Should be a 1D array.
    color_map=['winter', 'summer'] : list, optional
        Color map specifier. Each element should be on ['spring', 
        'autumn', 'cool', 'hot', 'winter', 'summer']. If not, must be 
        imported from matplotlib.cm
    
    Returns
    -------
    nothing
    
    Yields
    ------
    matplotlib.figure
    
    Raises
    ------
    "X should be array-like" : TypeError
        If len(X) raises error.
    "Y should be array-like" : TypeError
        If len(Y) raises error.
    "Z should be a 2D array" : TypeError
        If Z is not a 2D array.
    "X2 should be array-like" : TypeError
        If len(X2) raises error.
    "Y2 should be array-like" : TypeError
        If len(Y2) raises error.
    "Z2 should be a 2D array" : TypeError
        If Z is not a 2D array.
    "color_map should be..." : TypeError
        If color_map is not a size 2 list.
    "Each color_map should be..." : TypeError
        If not all color_map's elements are not an allowed string or 
        matplotlib colormap.
    
    """
    
    try:
        len(X)
    except:
        raise TypeError("X should be array-like")
    try:
        len(Y)
    except:
        raise TypeError("Y should be array-like")        
    try:
        Z[:,0]
    except:
        raise TypeError("Z should be a 2D array")
    try:
        Z2[:,0]
    except:
        raise TypeError("Z2 should be a 2D array")
    if X2 is not None:
        try:
            len(X2)
        except:
            raise TypeError("X2 should be array-like")
    if Y2 is not None:
        try:
            len(Y2)
        except:
            raise TypeError("Y2 should be array-like")     
    
    key = ['winter', 'summer', 'autumn', 'spring', 'cool', 'hot']
    try:
        color_map[1]
    except:
        raise TypeError("color_map should be a list of length 2")
    for cm in color_map:
        if cm not in key:
            if not isinstance(cm,
                              LinearSegmentedColormap):
                raise TypeError("Each color_map should be a matplotlib",
                                "colormap or should be on ",
                                key)

    X, Y = np.meshgrid(X, Y)
    if X2 is not None:
        if Y2 is not None:
            X2, Y2 = np.meshgrid(X2, Y2)
        else:
            X2, Y2 = np.meshgrid(X2, Y)
    elif Y2 is not None:
        X2, Y2 = np.meshgrid(X, Y2)
    else:
        X2, Y2 = X, Y
        
    fig = plt.figure(figsize=(13,6),tight_layout=True)  
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X, Y, Z, cmap = color_map[0])
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X2, Y2, Z2, cmap = color_map[1])
    fig.show()

#%%

def animation_2D(X, Y, figure_id=None, new_figure=True, 
                 frames_number=30,
                 label_function=lambda i : '{:.0f}'.format(i)):
    """Makes a series of 2D plots into an animation.
    
    Parameters
    ----------
    X : np.array
        Data's X values. Can be a 1D array of len X_len or a 2D array of 
        size (n, X_len), where each column corresponds to a different 
        data series.
    Y : np.array
        Data's X values. Can be a 1D array of len Y_len or a 2D array of 
        size (n, Y_len), where each column corresponds to a different 
        data series.
    figure_id=None : int, optional
        A matplotlib.pyplot.figure's ID. If specified, it plots on that 
        figure, which can be already formatted with axis labels, i.e.
    new_figure=False : bool, optional
        Indicates whether to make a new figure or not when 
        figure_id=None.
    frames_number=30 : int, optional
        Animation's number of frames.
    label_function : function, optional.
        Function that assigns frames' labels. Must take in one int 
        parameter and return one string.
    
    Returns
    -------
    animation : matplotlib.animation object
        Animation.

    Raises
    ------
    "Y should be a 2D array" : TypeError
        If Y doesn't have more than 1 column.
    "X, Y should have the same number of columns" : IndexError
        If X is not 1D and X, Y don't have the same number of columns.
    "X, Y should have the same number of rows" : IndexError
        If X, Y don't have the same number of rows.
    
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
        raise IndexError("X, Y should have the same number of columns")
    
    if X_rows != Y_rows:
        raise IndexError("X, Y should have the same number of rows")
    
    if figure_id is not None:
        fig = plt.figure(figure_id)
    elif new_figure:
        fig = plt.figure()
    else:
        fig = plt.gcf()

    try:
        ax = fig.axes
        ax[0]
    except IndexError:
        ax = [plt.axes()]
    
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

#%%

def animation_3D(X, Y, Z, figure_id=None, new_figure=True,
                 color_map='winter', frames_number=30,
                 label_function=lambda i : '{:.0f}'.format(i)):
    """Makes a series of 3D plots into an animation.
    
    Parameters
    ----------
    X : np.array
        Data's X values. Can be a 1D n-array of length X_len or a 2D 
        array of size (T, X_len), where each row corresponds to a 
        different data series.
    Y : np.array
        Data's X values. Can be a 1D n-array of length Y_len or a 2D 
        array of size (T, Y_len), where each row corresponds to a 
        different data series.
    Z : np.array
        Data. Must be a 3D array of size (T, Y_len, X_len).
    figure_id=None : int, optional
        A matplotlib.pyplot.figure's ID. If specified, it plots on that 
        figure, which can be already formatted with axis labels, i.e.
    new_figure=False : bool, optional
        Indicates whether to make a new figure or not when 
        figure_id=None.
    color_map='winter' : str, optional
        Color map specifier. Each element should be on ['spring', 
        'autumn', 'cool', 'hot', 'winter', 'summer']. If not, must be 
        imported from matplotlib.cm
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
    
    X, Y = np.meshgrid(X, Y)
        
    if figure_id is not None:
        fig = plt.figure(figure_id)
    elif new_figure:
        fig = plt.figure()
    else:
        fig = plt.gcf()
    #ax = axes(xlim=(x0,xf), ylim=(amin(Y),amax(Y)))
    ax = fig.gca(projection='3d')
        
    def update(i):
        
        ax.clear()
        
        ax.plot_surface(X, Y, Z[i,:,:], rstride=1, cstride=1, 
                        cmap=color_map, linewidth=0, antialiased=False)
        ax.set_zlim(np.amin(Z),np.amax(Z))
        ax.text(0, -2, 0.40, label_function(i), transform=ax.transAxes)
        
        plt.show()
        
        return
    
    anim = animation.FuncAnimation(fig, update, frames=frames_number, 
                                   interval=210)

    return anim;

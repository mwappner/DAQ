# -*- coding: utf-8 -*-
"""
This module contains some generic tools.

@author: GrupoFWP
"""

import numpy as np

#%%

def find(L, L0):
    """Takes an array of data and searches the index(es) of value L0.
    
    Parameters
    ----------
    L : list, np.array
        Array where I search
    L0 : int, float
        Value I search
    
    Returns
    -------
    ind : list
        Indexes of L that match L0.
    
    Raises
    ------
    "L must be a list or similar" : TypeError
        If L is not of an allowed type.
    
    """

    if not isinstance(L, list):
        try:
            L = list(L)
        except TypeError:
            L = [].append(L)
        else:
            return TypeError("L must be a list or similar")
            
    ind = []
    N = -1
    while N < len(L):
        val = L[N+1 : len(L)]
        try:
            # Write the index on L and not on val
            N = val.index(L0) + len(L) - len(val)
        except ValueError:
            break
        ind.append(N)
        
    return ind

#%%

def clip_between(value, lower=0, upper=100):
    '''Clips value to the (lower, upper) interval, i.e. if value
    is less than lower, it return lower, if its grater than upper,
    it return upper, else, it returns value unchanged.'''
    value = max(lower, value)
    value = min(upper, value)
    return value

#%%

def clear_queue(queue):
    """Clears a queue and returns all elements erased"""
    
    d = []
    while not queue.empty():
        data = queue.get()
        d.append(data)
    
    return d

#%%

def zeros(size, dtype=np.float64):
    
    """Analog to np.zeros but reshapes to N if size=(1, N)"""
    
    try:
        len(size)
        size = tuple(size)
    except TypeError:
        pass
    
    if isinstance(size, tuple):
        if size[0] == 1:
            size = size[1]
    
    return np.zeros(size, dtype=dtype)

def multiappend(nparray, new_nparray, fast_speed=True):
    
    """Analog to np.append but with 2D np.arrays"""
    
    try:
        nrows = len(new_nparray[:,0])
    except IndexError:
        nrows = 1
    if not fast_speed:
        try:
            nrows0 = len(np.nparray[:,0])
        except IndexError:
            nrows = 1
        if nrows0 != nrows:
            raise IndexError("Different number of rows.")
    
    if not nparray:
        return new_nparray
    
    elif nrows == 1:
        return np.append(nparray, new_nparray)
    
    else:
        construct = []
        for i in range(nrows):
            row = nparray[i,:]
            row = np.append(row, new_nparray[i,:])
            construct.append(row)
        return np.array(construct)

#%%

class TypedList(list):
    
    """A list that only appends a certain type"""    
    
    def __init__(self, Class):
        self.instance = Class

    def append(self, item):
        if not isinstance(item, self.instance):
            raise TypeError('Can only append {}'.format(self.instance))
        super().append(item)

# Simple, but apparently not the best way...
# https://stackoverflow.com/questions/3487434/overriding-append-method-after-inheriting-from-a-python-list

class NotCertainTypeList(list):
    
    """A list that doesn't append a certain type of elements"""
    
    def __init__(self, *iterable, show_exceptions=True):
        super().__init__(iterable)
        self.show_exceptions = show_exceptions
    
    def append(self, item):
        if not isinstance(item, self.instance):
            super().append(item)
        else:
            if self.show_exceptions:
                raise TypeError("Can't append {}".format(
                        self.instance))

#%%

class ObjectView(object):
    
    def __init__(self, d):
        
        self.__dict__ = d

class ObjectDict(dict):
    
    def __getattr__(self, name):
        
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        
        self[name] = value

    def __delattr__(self, name):
        
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

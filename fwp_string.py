# -*- coding: utf-8 -*-
"""
This module contains tools for dealing with strings.

Some of its most useful tools are:

value_error : function
    Rounds up value and error of a measure. Also makes a latex string.
find_numbers : function
    Returns a list of numbers found on a given string.

@author: Vall
"""

import re
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

def find_numbers(string):
    
    """Returns a list of numbers found on a given string
    
    Parameters
    ----------
    string: str
        The string where you search.
    
    Returns
    -------
    list
        A list of numbers (each an int or float).
    
    Raises
    ------
    "There's no number in this string" : TypeError
        If no number is found.

    """
    
    numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", string)
    
    if not numbers:
        raise TypeError("There's no number in this string")
    
    for i, n in enumerate(numbers):
        if '.' in n:
            numbers[i] = float(n)
        else:
            numbers[i] = int(n) 
    
    return numbers

def find_1st_number(string):
    
    """Returns the first float or int number of a string
    
    Parameters
    ----------
    string: str
        The string where you search.
    
    Returns
    -------
    number: int, float
        The number you found.
    
    Raises
    ------
    "There's no number in this string" : TypeError
        If no number is found.

    """
    
    number = find_numbers(string)[0]
    
    return number

#%%

def counting_sufix(number):
    
    """Returns a number's suffix string to use for counting.
    
    Parameters
    ----------
    number: int, float
        Any number, though it is designed to work with integers.
    
    Returns
    -------
    ans: str
        A string representing the integer number plus a suffix.
    
    Examples
    --------
    >> counting_sufix(1)
    '1st'
    >> counting_sufix(22)
    '22nd'
    >> counting_sufix(1.56)
    '2nd'
    
    """
    
    number = round(number)
    unit = int(str(number)[-1])
    
    if unit == 1:
        ans = 'st'
    if unit == 2:
        ans = 'nd'
    if unit == 3:
        ans = 'rd'
    else:
        ans = 'th'
    
    return ans

#%%

def string_recognizer(string, partial_keys,
                      priorized_marker="&"):
    """This is a function that recognizes a string from partial keys.
    
    Parameters
    ----------
    string : str
        Word to be recognized.
    partial_keys : dic
        Dictionary used for recognizing. Its keys should be str or tuple.
    
    Returns
    -------
    dynamic
    
    Examples
    --------
    >> partial_keys = {'a' : 1,
                       ('b','c') : 2,
                       ('&', 'r') : 3,
                       ('&', 'li', 'le') : 4}
    >> string_recognizer('a', partial_keys)
    1
    >> string_recognizer('add', partial_keys)
    1
    >> string_reconizer('b', partial_keys)
    2
    >> string_reconizer('c', partial_keys)
    2
    >> string_recognizer('rat', partial_keys)
    3
    >> string_recognizer('lion', partial_keys)
    4
    >> string_reconizer('let', partial_keys)
    4
    >> string_recognizer('ale', partial_keys)
    4
    """
    
    # Obviously, if 'string' is already on the values of partial_keys...    
    if string in partial_keys.values():
        return string
    string = string.lower()
    
    # OK then! First of all, I force all keys to be tuple
    non_priorized_keys = {}
    for key, value in partial_keys.items():
        if not isinstance(key, tuple):
            non_priorized_keys.update({tuple([key]) : value})
        else:
            non_priorized_keys.update({key : value})
    
    # Next I priorize some keys
    priorized_keys = {}
    for key in non_priorized_keys.keys():
        for element in list(key):
            if priorized_marker in element:
                priorized_keys.update({key : non_priorized_keys[key]})
                break
    
    # Then I define only the others as non-priorized
    for key in priorized_keys.keys():
        non_priorized_keys.pop(key)
    
    # Now I check the priorized keys are exclusive
    for examined_key in priorized_keys.keys():
        for other_key in priorized_keys.keys():
            
            if other_key is not examined_key:
            
                for element in examined_key:
                    if element is priorized_marker:
                        continue                    
                    if element in list(other_key):
                        return KeyError(
                                "Priorized keys are not exclusive",
                                "because {} and {}".format(
                                        element,
                                        other_key))
    
    # Next I check non-priorized keys are exclusive
    for examined_key in non_priorized_keys.keys():
        for other_key in non_priorized_keys.keys():
            
            if other_key is not examined_key:
            
                for element in examined_key:
                    if element in list(other_key):
                        return KeyError("Non-Priorized keys are ",
                                        "not exclusive because ",
                                        "{} and {}".format(
                                                element,
                                                other_key))

    for key, value in priorized_keys.items():
        for a_key in list(key):
            if a_key in string:
                return value

    for key, value in non_priorized_keys.items():
        for a_key in list(key):
            if a_key in string:
                return value

#%%

def append_data_to_string(*args):
    out = ''
    for value in args:
        out += '\t' + str(value)
    return out + '\n'
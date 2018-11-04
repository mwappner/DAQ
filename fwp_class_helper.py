# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 18:45:10 2018

@author: Usuario
"""

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

#%%

class ClassWithInstances:
    """Example of a class which allows dot calling instances.
    
    Examples
    --------
    >> class MyClass:
        def __init__(self, value=10):
            self.sub_prop = value
    >> instance_a, instance_b = MyClass(), MyClass(12)
    >> Z = ClassWithInstances(dict(a=instance_a,
                                   b=instance_b))
    >> Z.prop
    'Sorry'
    >> Z.prop = 'Not sorry'
    >> Z.prop
    'Not sorry'
    >> Z.a.sub_prop
    10
    >> Z.a.sub_prop = 30
    >> Z.a.sub_prop
    30
    >> instance_c = Z
    >> Z.update({'c': instance_c})
    >> Z.c.a.sub_prop
    30
    
    """
    
    def __init__(self, class_dict):
        
        self.__dict__.update(class_dict)
        self.prop = 'Sorry'
    
    @property
    def prop_2(self):
        return self.__prop_2
    
    @prop_2.setter
    def prop_2(self, value):
        self.__prop_2 = value
    
    def __getattr__(self, name):
        
        name = name.split('.')
        
        if len(name) == 1:
            
            return eval('self.{}'.format(name[0]))
        
        else:
            
            class_name = name[0]        
            attribute_name = name[-1]
            
            if class_name not in self.__dict__.keys():
                raise AttributeError("No such attribute: " + name)
            
            command = "self.__dict__['{}']".format(class_name)
            command = command + '.' + attribute_name
            
            return eval(command)

    def __setattr__(self, name, value):
        
        name = name.split('.')
        
        if len(name) == 1:

            self.__dict__.update({name[0] : value})
        
        else:
        
            class_name = name[0]        
            attribute_name = name[-1]
            
            command = "self.__dict__['{}']".format(class_name)
            command = command + '.' + attribute_name
            command = command + '=' + str(value)
            
            eval(command)
    
    def update(self, new_dict):
        
        self.__dict__.update(new_dict)
        
#%%

class InstancesDic:
    """Example of a class that holds a callable dictionary of instances.

    Examples
    --------
    >> class MyClass:
        def __init__(self, value=10):
            self.sub_prop = value
    >> instance_a, instance_b = MyClass(), MyClass(12)
    >> instance_c = ClassWithInstances(dict(a=instance_a,
                                            b=instance_b))
    >> Z = InstancesDic({1: instance_a,
                         2: instance_b,
                         3: instance_c})
    >> Z(1)
    <__main__.MyClass at 0x2e5572a2dd8>
    >> Z(1).sub_prop
    10
    >> Z(1).sub_prop = 30
    >> Z(1).sub_prop
    >> Z(3).b.sub_prop
    12
    >> Z(1,2)
    [<__main__.MyClass at 0x2e5573cfb00>, 
    <__main__.MyClass at 0x2e5573cf160>]
    
    Warnings
    --------
    'Z(1,2).prop' can't be done.
    
    """
    
    
    def __init__(self, dic):#, methods):
        
        self.__dict__.update(dic)
    
    def __call__(self, *key):

        if len(key) == 1:
            return self.__dict__[key[0]]
        
        else:
            return [self.__dict__[k] for k in key]

#    def __getattr__(self, name):
#        
#        if name in self.__methods:
#            values = {}
#            for k in self.__dic.__dict__:
#                v = eval("self.__dict__['{}'].{}".format(k, name))
#                values.update({k : v})
#        
#        return values
#
#    def __setattr__(self, name, value):
#        
#        if name in self.__methods:
#            for k in self.__dic.keys():
#                eval("self.__dict__['{}'].{} = {}".format(
#                        k, 
#                        name,
#                        value))
                
    def update(self, dic):
        
        self.__dict__.update(dic)
    
    def is_empty(self, key):
        
        if key in self.__dict__.keys():
            return False
        else:
            return True
        
#%%

class TypedList(list):
    """A list that only appends a certain type"""    
    
    def __init__(self, Class):
        self.instance = Class

    def append(self, item):
        if not isinstance(item, self.instance):
            raise TypeError('Nop because {}'.format(self.instance))
        super(TypedList, self).append(item)

# Simple, but apparently not the best way...
# https://stackoverflow.com/questions/3487434/overriding-append-method-after-inheriting-from-a-python-list

class NoNoneList(list):
    """A list that doesn't append None elements"""
    
    def append(self, item):
        if item is not None:
            super(NoNoneList, self).append(item)

#%%

class MethodRunOnList:
    """"A class that runs its method on a list"""
    
    def __init__(self, alist):#, methods):
        
        self.__list__ = alist
    
    def method(self, argument):
        
        result = []
        for l in self.__list__:
            result.append(argument + l)
        return result

#%%

class MyClass:
    
    def __init__(self, value=10):
        self.sub_prop = value
        self._prop = value
    
    @property
    def prop(self):
        return self._prop
    
    @prop.setter
    def prop(self, value):
        return self._prop
        
    def sub_method(self, item):
        return item * self.sub_prop
    
    def method(self, item):
        return item * self.prop

class SupraMyClass(MyClass):
    """A subclass that applies methods to a list of classes.
    
    Examples
    --------
    >> class MyClass:
    
        def __init__(self, value=10):
            self.sub_prop = value
            self._prop = value
        
        @property
        def prop(self):
            return self._prop
        
        @prop.setter
        def prop(self, value):
            return self._prop
            
        def sub_method(self, item):
            return item * self.sub_prop
        
        def method(self, item):
            return item * self.prop
        
    >> Z = SupraMyClass([MyClass(), MyClass(2)])
    >> Z.prop
    [10, 2]
    >> Z._prop
    [10, 2]
    >> Z.sub_prop
    [10, 2]
    >> Z.method(1)
    [10, 2]
    
    Warnings
    --------
    >> Z.prop
    [10, 2]
    >> Z.prop = 3
    >> Z.prop # I would like it to return [3, 3]
    3
    >> [l.prop for l in Z.__list__] # I'd also like [3, 3]
    [10, 2]
    
    """
    

    def __init__(self, alist):
        
        self.mylist = alist
    
    @property
    def mylist(self):
        return self.__list__
    
    @mylist.setter
    def mylist(self, value):
        self.__list__ = value
    
    def __getattr__(self, name):
        
        result = []
        for l in self.mylist:
            result.append(eval('l.{}'.format(name)))
        return result
    
#    def __setattr__(self, name, value):
#        
#        eval('self.{} = value'.format(name))

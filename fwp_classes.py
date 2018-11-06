# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 18:45:10 2018

@author: Usuario
"""

#%%       

class DynamicList(list):
    
    """Subclass that initializes a callable list
    
    Examples
    --------
    >>> a = DynamicList([1,2,3])
    >>> a(0, 1)
    [1,2]
    >> a(0,2)
    [1,3]
    >>> a.append(4)
    >>> a
    [1,2,3,4]
    
    """
    
    def __init__(self, iterable=[]):
        super().__init__(iterable)
    
    def __call__(self, *index):
        
        if len(index) == 1:
            return self[index[0]]
        
        else:
            return [self[i] for i in index]
    
#%%

class DynamicDict(dict):
    
    """Subclass that initializes a callable dictionary.
    
    Examples
    --------
    >>> a = DynamicDict()
    >>> a.update({'color': 'blue', 'age': 22})
    >>> a
    {'age': 22, 'color': 'blue'}
    >>> a('age', 'color')
    [22, 'blue']
    
    """
    
    def __init__(self, **elements):
        
        super().__init__(**elements)
        return
    
    def __call__(self, *key):

        if len(key) == 1:
            return self[key[0]]
        
        else:
            return [self[k] for k in key]
    
    def is_empty(self, key=None):
        
        if key is None:
            return len(list(self.keys())) == 0
        elif key in self.keys():
            return False
        else:
            return True  

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

#class ClassWithInstances:
#    
#    """Example of a class which allows dot calling instances.
#    
#    Examples
#    --------
#    >> class MyClass:
#        def __init__(self, value=10):
#            self.sub_prop = value
#    >> instance_a, instance_b = MyClass(), MyClass(12)
#    >> Z = ClassWithInstances(dict(a=instance_a,
#                                   b=instance_b))
#    >> Z.prop
#    'Sorry'
#    >> Z.prop = 'Not sorry'
#    >> Z.prop
#    'Not sorry'
#    >> Z.a.sub_prop
#    10
#    >> Z.a.sub_prop = 30
#    >> Z.a.sub_prop
#    30
#    >> instance_c = Z
#    >> Z.update(c=instance_c)
#    >> Z.c.a.sub_prop
#    30
#    
#    """
#    
#    def __init__(self, class_dict):
#        
#        self.__dict__.update(class_dict)
#        self.prop = 'Sorry'
#    
#    @property
#    def prop_2(self):
#        return self.__prop_2
#    
#    @prop_2.setter
#    def prop_2(self, value):
#        self.__prop_2 = value
#    
#    def __getattr__(self, name):
#        
#        name = name.split('.')
#        
#        if len(name) == 1:
#            
#            return eval('self.{}'.format(name[0]))
#        
#        else:
#            
#            class_name = name[0]        
#            attribute_name = name[-1]
#            
#            if class_name not in self.__dict__.keys():
#                raise AttributeError("No such attribute: " + name)
#            
#            command = "self.__dict__['{}']".format(class_name)
#            command = command + '.' + attribute_name
#            
#            return eval(command)
#
#    def __setattr__(self, name, value):
#        
#        name = name.split('.')
#        
#        if len(name) == 1:
#
#            self.__dict__.update({name[0] : value})
#        
#        else:
#        
#            class_name = name[0]        
#            attribute_name = name[-1]
#            
#            command = "self.__dict__['{}']".format(class_name)
#            command = command + '.' + attribute_name
#            command = command + '=' + str(value)
#            
#            eval(command)
#    
#    def update(self, **new_dict):
#        
#        self.__dict__.update(dict(new_dict))

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

class WrapperList(list):
    
    """A list subclass that applies methods to a list of instances.
    
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
            self._prop = value
            
        def sub_method(self, item):
            return item * self.sub_prop
        
        def method(self, item):
            return item * self.prop
        
    >> Z = WrapperList([MyClass(), MyClass(2)])
    >> Z.prop
    [10, 2]
    >> Z._prop
    [10, 2]
    >> Z.sub_prop
    [10, 2]
    >> Z.method(2)
    [20, 4]
    >> Z.prop = 3
    >> Z.prop
    [3, 3]
    >> Z.append(MyClass(1))
    >> Z.prop
    [3, 3, 1]
    >> Z.prop = [10, 2, 1]
    >> Z.prop
    [10, 2, 1]
    
    Warnings
    --------
    >> Z.prop = [2, 3]
    >> Z.prop
    [[2,3], [2,3], [2,3]]
    
    """

    def __init__(self, iterable=[]):
        
        super().__init__(iterable)

    def __transf_methods__(self, methods_list):
        
        def function(*args, **kwargs):
            results = [m(*args, **kwargs) for m in methods_list]
            return results
        
        return function

    def __getattr__(self, name):
        
        if name in dir(self):
        
            super().__getattribute__(name)
        
        else:
            
            result = []
            for ins in self:
                result.append(ins.__getattribute__(name))
            if callable(result[0]):
                result = self.__transf_methods__(result)
            return result
    
    def __setattr__(self, name, value):
        
        if isinstance(value, list) and len(value) == len(self):
            for ins, v in zip(self, value):
                ins.__setattr__(name, v)
        else:
            for ins in self:
                ins.__setattr__(name, value)

#%%

class WrapperDict(dict):
    
    """A dict subclass that applies methods to a dict of instances.
    
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
            self._prop = value
            
        def sub_method(self, item):
            return item * self.sub_prop
        
        def method(self, item):
            return item * self.prop
        
    >> Z = WrapperDict(a=MyClass(), b=MyClass(2))
    >> Z.prop
    {'a': 10, 'b': 2}
    >> Z.update(c=MyClass(1))
    >> Z.prop
    {'a': 10, 'b': 2, 'c': 1}
    >> Z.method(2)
    {'a': 20, 'b': 4, 'c': 2}
    >> Z.prop = 3
    >> Z.prop
    {'a': 3, 'b': 3, 'c': 3}
    >> Z.prop = {'a': 10, 'b': 2, 'c': 1}
    >> Z.prop
    {'a': 10, 'b': 2, 'c': 1}
    
    Warnings
    --------
    >> Z.prop = [2, 3]
    >> Z.prop
    [[2,3], [2,3], [2,3]]
    
    """

    def __init__(self, **elements):
        
        super().__init__(**elements)
#        self.__update__ = super().update
    
    def __transf_methods__(self, methods_dic):
        
        def function(*args, **kwargs):
            results = {}
            for key, method in methods_dic.items():
                results.update({key: method(*args, **kwargs)})
            return results
        
        return function
    
    def __getattr__(self, name):
        
        if name in dir(self):
            super().__getattribute__(name)        
        else:
            result = {n : ins.__getattribute__(name) 
                      for n, ins in self.items()}
            if callable(list(result.values())[0]):
                result = self.__transf_methods__(result)
            return result
    
    def __setattr__(self, name, value):
        
        if isinstance(value, dict):
            for ins, v in zip(self.values(), value.values()):
                ins.__setattr__(name, v)
        else:
            for ins in self.values():
                ins.__setattr__(name, value)

#%%

class ClassWithInstances:
    
    """Example of a class which allows dot calling instances.
    
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
            self._prop = value
            
        def sub_method(self, item):
            return item * self.sub_prop
        
        def method(self, item):
            return item * self.prop
        
    >> Z = ClassWithInstances(a=MyClass(), b=MyClass(2))
    >>
    >> # Let's check dot calling
    >> Z.a.sub_prop 
    10
    >> Z.a.sub_prop = 1
    >> Z.a.sub_prop
    1
    >>
    >> # Let's check dot calling all instances at once
    >> Z.sub_prop
    {'a': 1, 'b': 2}
    >> Z.all.sub_prop = 3
    >> Z.sub_prop
    {'a': 3, 'b': 3}
    >> Z.all.sub_prop = {'a': 1}
    >> Z.sub_prop
    {'a': 1, 'b': 3}
    >> 
    >> # This also works with methods
    >> Z.a.prop
    10
    >> Z.a.method(2)
    20
    >> Z.all.prop
    {'a': 10, 'b': 2}
    >> Z.all.method(2)
    {'a': 20, 'b': 4}
    
    Warnings
    --------
    >> # There are some wrong ways of calling things
    >> Z.all.method()
    {'a': 20, 'b': 4}
    >> Z.method()
    RecursionError: maximum recursion depth exceeded
    >> Z.all.prop
    {'a': 10, 'b': 2}
    >> Z.prop
    RecursionError: maximum recursion depth exceeded
    >>
    >> # Has some updating problems
    >> Z.update(c=MyClass(3))
    >> Z.sub_prop
    {'a': 1, 'b': 2, 'c': 3}
    >> Z.all.sub_prop
    {'a': 1, 'b': 3}
    
    """
    
    def __init__(self, **instances):
        
        self.__instances = {}
        self.__properties = {}
        self.update(**instances)
        self.all = WrapperDict(**self.instances)
    
    @property
    def instances(self):
        return self.__instances
    
    @instances.setter
    def instances(self, value):
        return AttributeError("See 'update' method instead.")
    
    @property
    def properties(self):
        return self.__properties
    
    @properties.setter
    def properties(self, value):
        return AttributeError("Shouldn't set this manually!")
    
    def update(self, **instances):
        
        instances = dict(instances)
        self.__dict__.update(instances)
        self.instances.update(instances)
        for name, ins in instances.items():
            self.properties.update({name: m for m in dir(ins)})
    
    def __getattr__(self, name):
        
        name = name.split('.')
        
        if len(name) == 1:
            
            name = name[0]
            
            if name in self.properties.values():

                results = {}
                for n, ins in self.instances.items():
                    try:
                        results.update({n : #n + '_' + name : 
                            eval("ins.{}".format(name))})
                    except:
                        pass                
                return results
                
            else:
            
                return eval('self.{}'.format(name))
                
        else:
            
            instance_name = name[0]        
            attribute_name = name[-1]
            
            if instance_name not in self.__dict__.keys():
                raise AttributeError("No such attribute: " + name)
            
            command = "self.__dict__['{}']".format(instance_name)
            command = command + '.' + attribute_name
            
            return eval(command)

    def __setattr__(self, name, value):
        
        name = name.split('.')
        
        if len(name) == 1:
            
            name = name[0]
            self.__dict__.update({name : value})
        
        else:
        
            class_name = name[0]        
            attribute_name = name[-1]
            
            command = "self.__dict__['{}']".format(class_name)
            command = command + '.' + attribute_name
            command = command + '=' + str(value)
            
            eval(command)
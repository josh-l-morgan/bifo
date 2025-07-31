# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 11:55:06 2025

@author: jlmorgan
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 18:18:23 2025

@author: jlmorgan
"""

import numpy as np
from types import SimpleNamespace
import time
            
def mat_struct_to_dict(obj):
    """Recursively convert MATLAB structs to nested Python dicts."""
    if isinstance(obj, np.ndarray):
        return [mat_struct_to_dict(o) for o in obj]
    elif hasattr(obj, '_fieldnames'):
        return {f: mat_struct_to_dict(getattr(obj, f)) for f in obj._fieldnames}
    else:
        return obj
  
    
def dict_to_namespace(d):
    """Recursively convert a dictionary (or struct-like object) to SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, (list, np.ndarray)) and all(isinstance(i, dict) for i in d):
        # Convert list of dicts to list of namespaces
        return [dict_to_namespace(i) for i in d]
    else:
        return d
   
   
def explore_namespace(ns, level=1, indent=1, name='root'):
    prefix = '  ' * indent
    if isinstance(ns, SimpleNamespace):
        for key, val in vars(ns).items():
            if isinstance(val, SimpleNamespace):
                print(f"{prefix}{key}:     (SimpleNamespace)")
                if level>1:
                    explore_namespace(val, level - 1, indent + 1)
            elif isinstance(val, np.ndarray):
                print(f"{prefix}{key}:     ndarray shape {val.shape}, dtype {val.dtype}")
            elif isinstance(val, list):
                print(f"{prefix}{key}:     list of {len(val)} elements")
            else:
                print(f"{prefix}{key}:     {type(val).__name__} = {val}")
    else:
        print(f": {ns}")          

    
class ticTocDic:
    """
        same as ticToc but with dictionary
        Example:
        t = ticToc()
        
        t.b('all')
        t.b(0) # begin timer
        something = 4
        t.e(0) # end timer
        
        t.b(1) 
        something_else = 5
        t.e(1)
        
        difference1 = t.t[2]
        
        t.p() # print differences
        
    """
    def __init__(self,num_t=10):
        self.t = np.zeros((num_t,4),'float64')
        self.td = {}
    
    def m(self,keys):
        for k in keys:
            self.td[k] = np.zeros(4,'float64')
            
    
    def b(self,t_num=0):
        if isinstance(t_num,str):
            self.td[t_num][0:2] = [time.time(),0]
        else:
            self.t[t_num,0:2] = [time.time(),0]

    def e(self,t_num=0):
        if isinstance(t_num,str):
            self.td[t_num][1] = time.time()
            self.td[t_num][2] = self.td[t_num][1] - self.td[t_num][0]
            self.td[t_num][3] = self.td[t_num][3] + self.td[t_num][2]
        else:
            self.t[t_num,1] = time.time()
            self.t[t_num,2] = self.t[t_num,1] - self.t[t_num,0]
            self.t[t_num,3] = self.t[t_num,3] + self.t[t_num,2]
    
    def s(self,sig_dig=2, tot=0):
        
        is_beginning = np.where(self.t[:,0]>0)[0]
        if tot:
            t_string = 'total times'
            t_idx = 3
        else:
            t_string = 'last times:'
            t_idx = 2
        cur_time = time.time()
        
        for k in self.td.keys():
            if self.td[k][0]:
                if self.td[k][1]:
                    t_string = t_string + f' {k} = {self.td[k][t_idx]:.{sig_dig}f}s,'
                else:
                    diff = cur_time - self.td[k][0]
                    t_string = t_string + f' {k} = {diff:.{sig_dig}f}s,'
            else:
                    t_string = t_string + f' {k} = none,'

        
        for v in is_beginning:
            if self.t[v][1]:
                t_string = t_string + f' #{v} = {self.t[v,t_idx]:.{sig_dig}f}s,'
            else:
                diff = cur_time - self.td[k][0]
                t_string = t_string + f' #{v} = {diff:.{sig_dig}f}s,'

            
        return t_string[:-1]
    
     
    def pl(self, sig_dig=2):
        print(self.s(sig_dig,tot=0))
        
    def pt(self, sig_dig=2):
        print(self.s(sig_dig,tot=1))
    
    def clear(self,t_num = None):
        if t_num is None:
            self.t = self.t * 0
            self.td = {}
        else:
            if isinstance(t_num,str):
                self.td.pop(t_num)
            else:
                self.t[t_num] = np.zeros(4, 'float32')
    
    
    
    
    
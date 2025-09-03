# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 15:24:23 2025

@author: jlmorgan
"""

#import napari


def goto(viewer, layer, z=None, y=None, x=None, level=None):
    # optionally change pyramid level (0 = finest, bigger = coarser)
    if level is not None and hasattr(layer, "data_level"):
        layer.data_level = int(level)

    nd = viewer.dims.ndim            # total axes in this scene
    # indices of Z,Y,X are always the last three axes in napari
    if nd >= 3 and z is not None:
        viewer.dims.set_point(nd - 3, int(z))
    if nd >= 2 and y is not None:
        viewer.dims.set_point(nd - 2, int(y))
    if nd >= 1 and x is not None:
        viewer.dims.set_point(nd - 1, int(x))



def limit_level(nv, layer, min_level):
        
    def pin_level(event=None, *, layer=layer, target=min_level):
        if getattr(layer, "data_level", None) is not None and layer.data_level < target:
            layer.data_level = target
    
    # set once, then keep re-applying whenever napari tries to change levels
    layer.data_level = min_level
    nv.camera.events.zoom.connect(pin_level)
    nv.dims.events.current_step.connect(pin_level)
    
    # optional: enforce again whenever the layer’s level changes
    #layer.events.data_level.connect(pin_level)











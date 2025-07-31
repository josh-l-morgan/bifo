# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 18:20:50 2025

@author: jlmorgan

tools for working with VAST

"""
import numpy as np

def vast_to_map_request(map_request,
        vast_location=(15163, 15397, 373),
        check_size=[4, 256, 256]
        ):
   
    fov_center = np.array(vast_location) 
    fov_center = fov_center[[2,1,0]]
    fov_corner_1 = fov_center - (np.array(check_size) * map_request['mip_level'] / 2) # find lower corner

    map_request["volume_lower_corner_zyx"] = fov_corner_1.astype(int)
    map_request['volume_size_zyx'] = check_size
    
    return map_request

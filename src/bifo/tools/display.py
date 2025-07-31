# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 15:08:05 2025

@author: jlmorgan
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap



def col_map(c_type='rand',num_col = 256):
    
    if c_type == 'rand':
        
        cmap_raw = plt.get_cmap('hsv',num_col)
        cmap_lin = cmap_raw(np.linspace(0,1,num_col))
        cmap_lin =cmap_lin[np.random.permutation(num_col)]
        cmap_arr = np.concatenate((np.array([[0, 0, 0, 1]]),cmap_lin),0)
        cmap = ListedColormap(cmap_arr)
        
    return cmap
    

class figure:
    def __init__(self, new=0, num_ax=1):
        
        if new:
            self.fig = plt.figure(figsize=(8,8))
        else:                      
            fig_num = plt.get_fignums()
            if fig_num:
                self.fig = plt.figure(fig_num[-1])
                self.fig.clear()
            else:
                self.fig = plt.figure(figsize=(8,8))
        
        self.ax = []
        ax_rows = np.floor(num_ax ** (1/2))
        ax_cols = np.ceil(num_ax/ax_rows)
        for a in range(num_ax):
            self.ax.append(self.fig.add_subplot(int(ax_rows), int(ax_cols), a+1))

    def update(self):
        self.fig.canvas.flush_events()
        self.fig.canvas.draw()
        plt.pause(0.001) 

    def make_img(self, ax_id=0, i_shape=[512, 512], vmin=0, vmax=255, cmap='grey'):
        blank = np.zeros(np.squeeze(i_shape))
        self.ax[ax_id].clear()
        self.ax[ax_id].img = self.ax[ax_id].imshow(
            blank, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
        
   
    
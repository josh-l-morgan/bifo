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
    def __init__(self, num_ax=1, fig_name=None, rows=None):
        
        self.fig = plt.figure(figsize=(8,8), num=fig_name)
        self.fig.clear() 
          
        self.ax = []
        if rows is None:
            ax_rows = np.floor(num_ax ** (1/2))
        else:
            ax_rows = rows
        ax_cols = np.ceil(num_ax/ax_rows)
        for a in range(num_ax):
            self.ax.append(self.fig.add_subplot(int(ax_rows), int(ax_cols), a+1))

    def update(self):
        for a in self.ax:
            if hasattr(a, 'img'):
                h, w = a.img.get_array().shape
                a.img.set_extent((0, w, h, 0))
        for a in self.ax:
            a.relim()
            a.autoscale_view()
        self.fig.canvas.flush_events()
        self.fig.canvas.draw()
        plt.pause(0.01) 

    def make_img(self, ax_id=0, i_shape=[512, 512], vmin=0, vmax=255, cmap='grey'):
        blank = np.zeros(np.squeeze(i_shape))
        self.ax[ax_id].clear()
        self.ax[ax_id].img = self.ax[ax_id].imshow(
            blank, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')


def raise_window(figname=None):
    """
    Raise the plot window for Figure figname to the foreground.  If no argument
    is given, raise the current figure.
    This function will only work with a Qt graphics backend.  It assumes you
    """

    if figname: plt.figure(figname)
    cfm = plt.get_current_fig_manager()
    cfm.window.activateWindow()
    cfm.window.raise_()
   
    
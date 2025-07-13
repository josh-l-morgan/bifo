# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 10:47:40 2025

@author: jlmorgan
"""

import os
import numpy as np
import re
from PIL import Image, ExifTags
import matplotlib.pyplot as plt

plt.ion()
fig = plt.figure
    
    
class fetchDiced:
    """
    Collect information required for fetching images from a dataset that 
    Has been diced to be red by VAST
    """    

    def __init__(self, mr):
        """
               
        Standard Example: 
            map_request = {
                "diced_dir": "//storage1.ris.wustl.edu/jlmorgan/Active/morganLab/DATA/LGN_Developing/KxR_P11LGN/diced/",
                "mip_level": 1,
                "analyze_section": [400], 
                "volume_lower_corner_zyx": [600,25000, 35000]
                "volume_size_zyx": [32, 1024, 1024],
                "chunk_size_zyx": [16, 256, 256],
                "chunk_overlap": [8, 128, 128],
                }
            
            fd = fetchDiced(map_request)
            for i in range(fd.num_chunk)
                fd.readChunk(i)
                fd.viewChunk() #optional display
                coolAnalysis(fd.chunk)
                
        Additional Data:
            sec_dirs: list of section names
            sec_nums: np.array of section numbers
            mappedMips: dictionary containing information about mips. produced by mapMipDir
            
        Additional Functions:
              mapMipDirs: identify mip folders for all sections. produce mappedMips   
        
        """
        
        self.map_request = mr
        self.diced_dir = mr["diced_dir"]
        
        self.analyze_sections = mr["analyze_section"]
        self.analyze_mip = [mr["mip_level"]]

        self.findSections()  
        #self.mapMipDirs()
        self.getTileMetadata()        
        self.fetchList(mr["volume_lower_corner_zyx"], mr["volume_size_zyx"])
        self.chunkList(mr["chunk_size_zyx"],mr["chunk_overlap"])
        self.prepareToReadChunks()
                
    def findSections(self):
        # read section list as firs subdirector
                 
        self.sec_dirs = sorted(
            [d.name for d in os.scandir(self.diced_dir) if d.is_dir() and d.name.isdigit()],
            key=int 
        )
        self.sec_nums = np.array([int(d) for d in self.sec_dirs]) 
        
        
    def mapMipDirs(self):
        # generate array showing wich sections have which mip level dirs
             
        sec_mip_dirs = []
        s_dir = []
        has_mip_array = np.zeros((len(self.sec_dirs),12),bool)
        
        for i,s in enumerate(self.sec_dirs):
           
            s_dir.append(self.diced_dir + s + "/") 
            mip_dirs = sorted(
                [d.name for d in os.scandir(s_dir[i]) if d.is_dir() and d.name.isdigit()],
                key=int 
            )
            sec_mip_dirs.append(mip_dirs)
            
            # record presence of mips
            if mip_dirs:
                mip_int = [int(m) for m in mip_dirs]
                has_mip_array[i, mip_int] = 1
                
        mip_presence = has_mip_array.sum(0)
        max_mip_number = mip_presence.max()
        mip_missing = max_mip_number - mip_presence
        self.mappedMips = {"sec_mip_bool": has_mip_array,
                           "total": max_mip_number,
                           "missing": mip_missing}
                
    def getTileMetadata(self):
        # analyze tiles at mip
        
        if self.analyze_sections == 'None':
           searchSections = np.random.permutation(self.sec_nums)
           tilEnd = 'false'
        else:
           searchSections = self.analyze_sections
           tilEnd = 'true'
           
        if self.analyze_mip == 'None':
           use_mip = 0
        else:
           use_mip = self.analyze_mip[0]
            
        
        self.img_info = []
        for i,s in enumerate(searchSections):
            sec_targ = int(np.where(self.sec_nums==s)[0][0])
            sec_mip_dir = self.diced_dir + self.sec_dirs[sec_targ] + f"/{use_mip}/" 
            if os.path.isdir(sec_mip_dir):    
                # parse tiles
                tile_names = [d.name for d in os.scandir(sec_mip_dir)]
                rc_list = [re.findall(r'\d+', s) for s in tile_names]
                rc_array = np.array([[int(c) for c in r] for r in rc_list])
                rc_max = rc_array.max(0)
                rc_mid = np.round(rc_max/2)
                rc_targ = int(np.where((rc_array[:,0] == rc_mid[0]) & (rc_array[:,1]==rc_mid[1]))[0][0])
                image_path = sec_mip_dir + tile_names[rc_targ]
                try:
                    img = Image.open(image_path)  
                except:
                    print(f"could not open {image_path}")
                else:
                    self.img_info.append({"size": img.size})
                    if not tilEnd:
                        break
                        
            self.tileSize = self.img_info[0]['size'][0]
                    
    def fetchList(self, raw_corner, volume_zyx, use_mip = 'None'):
        
    
        
        if use_mip == 'None':
            use_mip = 0
        
        tileSize = self.tileSize
        
        # get voxel corners
        corner1 = np.array(raw_corner)
        corner1[1:2] = np.round(corner1[1:2]/2)
        corner2 = corner1 + np.array(volume_zyx)
        
        # get tile corners
        cornerTile1 = corner1.copy();
        cornerTile1[1:] = np.floor(cornerTile1[1:] / tileSize)
        cornerTile2 = corner2.copy();
        cornerTile2[1:] = np.floor(cornerTile2[1:] / tileSize) 
        
        # make tile list
        tile_zrc_list = []
        for r in range(cornerTile1[1], cornerTile2[1]+1):
            for c in range(cornerTile1[2], cornerTile2[2]+1):
                for z in range(cornerTile1[0], cornerTile2[0]+1):
                    tile_zrc_list.append([z,r,c])
                    
        # make tile names
        tile_paths = []
        for i,zrc in enumerate(tile_zrc_list):
             tile_paths.append(f"{self.diced_dir}{zrc[0]}/{use_mip}/{zrc[1]}_{zrc[2]}.png")
                 
        fetch_list = {
            "tile_zrc_array": np.array(tile_zrc_list),
            "tile_paths": tile_paths,   
            "corner1": corner1,
            "corner2": corner2
        }
        
        self.fetch_list = fetch_list
    
    def chunkList(self,raw_chunck_size,raw_overlap):
        
        #note, most for loops should be replaced by array operations
        
        fl = self.fetch_list
        tileSize = self.tileSize
        chunk_size = np.array(raw_chunck_size)
        overlap = np.array(raw_overlap)
        list_ok = 'true'
        
        step = chunk_size - overlap
        
        # make tile list
        chunk_zrc_corner1 = []
        for r in range(fl['corner1'][1], fl['corner2'][1]-step[1]+1,step[1]):
            for c in range(fl['corner1'][2], fl['corner2'][2]-step[2]+1,step[2]):
                for z in range(fl['corner1'][0], fl['corner2'][0]-step[0]+1,step[0]):
                    chunk_zrc_corner1.append([z,r,c])
        
        num_chunk = len(chunk_zrc_corner1)
        corners1 = np.array(chunk_zrc_corner1)
        corners2 = corners1 + chunk_size - 1
        
        cornersTile1 = np.copy(corners1)
        cornersVox1 = np.copy(corners1)
        cornersTile1[:,1:] = np.floor(corners1[:,1:] / tileSize)
        cornersVox1[:,1:] = np.remainder(corners1[:,1:], tileSize) #0 to tileSize-1
        
        first_tile_vox_num = np.minimum(chunk_size[1]-1,tileSize-cornersVox1-1)
        
               
        cornersTile2 = np.copy(corners2)
        cornersVox2 = np.copy(corners2)
        cornersTile2[:,1:] = np.floor(corners2[:,1:] / tileSize)
        cornersVox2[:,1:] = np.remainder(corners2[:,1:], tileSize)
        
        # create four tile -> chunck index mappings for each chunk, chunk quarters
        chunk_vox = np.zeros([num_chunk,2,2,2,2],int)
        tile_vox = np.zeros([num_chunk,2,2,2,2],int)
        tile_rc = np.zeros([num_chunk,2,2,2],int)
        tile_zs = np.zeros([num_chunk,chunk_size[0]],int)
        
        # Arange tile r c information into quandrants
        tile_rc[:,0,:,0] = cornersTile1[:,1][:, None] * np.ones((1, 2))
        tile_rc[:,:,0,1] = cornersTile1[:,2][:, None] * np.ones((1, 2))
        tile_rc[:,1,:,0] = cornersTile2[:,1][:, None] * np.ones((1, 2))
        tile_rc[:,:,1,1] = cornersTile2[:,2][:, None] * np.ones((1, 2))
    
        # negate quandrants where chunck doesnt cross tiles
        same_r = tile_rc[:,0,0,0] == tile_rc[:,1,0,0]
        same_c = tile_rc[:,0,0,1] == tile_rc[:,0,1,1]
        tile_rc[same_r,1,:,:] = -1 
        tile_rc[same_c,:,1,:] = -1
    
        # get z positions for each chunk
        tile_zs = corners1[:,0][:,None] + np.arange(0,chunk_size[0])[None,:]  
    
        # quarter r_q = 0, c_q = 0 
        chunk_vox[:,0,0,0,:] = first_tile_vox_num[:,1][:,None] * [0,1] 
        chunk_vox[:,0,0,1,:] = first_tile_vox_num[:,2][:,None] * [0,1] 
        # quarter r_q = 0, c_q = 1 
        chunk_vox[:,0,1,0,:] = first_tile_vox_num[:,1][:,None] * [0,1] 
        chunk_vox[:,0,1,1,:] = first_tile_vox_num[:,2][:,None] * [1,0] + [1,0] + [0,chunk_size[1]-1]
        # quarter r_q = 1, c_q = 0 
        chunk_vox[:,1,0,0,:] = first_tile_vox_num[:,1][:,None] * [1,0] + [1,0] + [0,chunk_size[1]-1]
        chunk_vox[:,1,0,1,:] = first_tile_vox_num[:,2][:,None] * [0,1] 
        # quarter r_q = 1, c_q = 0 
        chunk_vox[:,1,1,0,:] = first_tile_vox_num[:,1][:,None] * [1,0] + [1,0] + [0,chunk_size[1]-1]
        chunk_vox[:,1,1,1,:] = first_tile_vox_num[:,2][:,None] * [1,0] + [1,0] + [0,chunk_size[1]-1]

        first_reach = cornersVox1 + first_tile_vox_num
        # quarter r_q =0 , c_q = 0  
        tile_vox[:,0,0,0,:] = np.stack([cornersVox1[:,1], first_reach[:,1]], axis = 1)
        tile_vox[:,0,0,1,:] = np.stack((cornersVox1[:,2], first_reach[:,2]), axis = 1)
        # quarter r_q =1 , c_q = 0  
        tile_vox[:,1,0,0,:] = cornersVox2[:,1][:,None] * [0,1]         
        tile_vox[:,1,0,1,:] = np.stack((cornersVox1[:,2], first_reach[:,2]), axis = 1) 
        # quarter r_q =0 , c_q = 0  
        tile_vox[:,0,1,0,:] = np.stack([cornersVox1[:,1], first_reach[:,1]], axis = 1)
        tile_vox[:,0,1,1,:] = cornersVox2[:,2][:,None] * [0,1] 
        # quarter r_q =0 , c_q = 0  
        tile_vox[:,1,1,0,:] = cornersVox2[:,1][:,None] * [0,1]
        tile_vox[:,1,1,1,:] = cornersVox2[:,2][:,None] * [0,1]
        
        # chunk_vox[:,:,:,:,1] = chunk_vox[:,:,:,:,1] +1 #stupid python indexing...
        # tile_vox[:,:,:,:,1] = tile_vox[:,:,:,:,1] +1 #stupid python indexing...

                               
                        
        # Test mapping logic, all references should add up to chunk size
        count_chunk_vox_rc = np.zeros([num_chunk,2])
        count_tile_vox_rc = np.zeros([num_chunk,2])
        count_same_t = np.zeros([num_chunk,2])
        for i in range(num_chunk):
            for d in range(2):
                for r_q in range(2):
                    for c_q in range(2):
            
                        if tile_rc[i,r_q,c_q,d]>0:
                            count_tile_vox_rc[i,d] = count_tile_vox_rc[i,d]  + tile_vox[i,r_q,c_q,d,1] - tile_vox[i,r_q,c_q,d,0] + 1
                            count_chunk_vox_rc[i,d] = count_chunk_vox_rc[i,d]  + chunk_vox[i,r_q,c_q,d,1] - chunk_vox[i,r_q,c_q,d,0] + 1
            
        bad_tile_count = np.sum(count_tile_vox_rc != (chunk_size[1]))   
        bad_chunk_count = np.sum(count_chunk_vox_rc != (chunk_size[1]))   
        
        if any(((bad_tile_count>0),(bad_chunk_count>0))):
            print('bad reference vox numbers')
            list_ok = 'false'
  
        # Find ids for tiles
        fl_zrc = fl['tile_zrc_array']
        tile_id = np.zeros([num_chunk,2,2,chunk_size[0]])-1 #tile id in quart space of chunck, row, colum, z
        for i in range(num_chunk):
            for r_q in range(2):
                for c_q in range(2):
                    is_r = (fl_zrc[:,1] == tile_rc[i,r_q,c_q,0])
                    is_c = (fl_zrc[:,2] == tile_rc[i,r_q,c_q,1])
                    for zs in range(chunk_size[0]):
                        is_z = fl_zrc[:,0] == tile_zs[i,zs]
                        targ_tile = np.where( is_z & is_r & is_c)[0]
                        if targ_tile.shape[0] == 0:
                            tile_id[i,r_q,c_q,zs] = -1 
                        else:
                            tile_id[i,r_q,c_q,zs] = targ_tile[0]
                            
        self.num_chunk = num_chunk                  
        self.chunk_list = {
            "chunk_size": chunk_size,
            "num_chunk": num_chunk,
            "chunk_vox": chunk_vox,
            "tile_vox": tile_vox,
            "tile_rc": tile_rc,
            "tile_id": tile_id,
            "list_ok": list_ok
        }
        
    def prepareToReadChunks(self):
            
        class tileCache:
            def __init__(self,md):
                
                csh_size = 2048
                self.size = csh_size; # how many tiles to hold in memory
                self.cache = ['' for i in range(csh_size)]
                self.tile_ids = np.zeros(csh_size,int)-1
                self.tile_paths = md.fetch_list['tile_paths']
                self.tile_age = np.zeros(csh_size)
                
            def requestTiles(self,need_tiles):
                
                self.tile_age = self.tile_age + 1
                existing_idx = np.where(np.isin(need_tiles,self.tile_ids))
                cache_id = np.zeros(need_tiles.shape[0])-1
                for i in range(need_tiles.shape[0]):
                    tile_targ = np.where(self.tile_ids == need_tiles[i])[0]
                    if tile_targ.shape[0] > 0:
                        cache_id[i] = tile_targ[0]
                
                # if we need to load new tiles
                still_need = np.where(cache_id<0)[0]

                if  still_need.shape[0]>0:
                    keep = np.isin(self.tile_ids,need_tiles)
                    loose = ~keep
                    
                    for sn in range(still_need.shape[0]):
                        tile_id = int(need_tiles[still_need[sn]])
                        oldest = np.where(loose & 
                           (self.tile_age == self.tile_age[loose].max()))[0][0]                    
                        tile_path = self.tile_paths[tile_id]
                        self.cache[oldest] = np.array(Image.open(tile_path))
                        self.tile_ids[oldest] = need_tiles[sn] # record new id for cache possition
                        self.tile_age[oldest] = 0
                        cache_id[still_need[sn]] = oldest
                
                self.cache_id = cache_id
                
        # Make cache and chunk array        
        self.tile_cache = tileCache(self)
        self.chunk = np.zeros(self.chunk_list['chunk_size'])
        
      
    def readChunk(self, chunk_id):
        
        # Grab information from chunk list for chunk_id chunk
        chunk_size = self.chunk_list['chunk_size']
        c_tile_rc = self.chunk_list['tile_rc'][chunk_id,:]
        c_tile_id = self.chunk_list['tile_id'][chunk_id,:]
        c_tile_vox = self.chunk_list['tile_vox'][chunk_id,:]
        c_chunk_vox = self.chunk_list['chunk_vox'][chunk_id,:]
        
        # get cache ids for tiles and ask requestTiles to load them into cache
        good_tile_mask = c_tile_id >= 0 #np.where(c_tile_id>=0,1,0)
        need_tiles = c_tile_id[good_tile_mask]
        self.tile_cache.requestTiles(need_tiles)
        cache_positions = c_tile_id.copy()
        cache_positions[good_tile_mask] = self.tile_cache.cache_id
        
             
        # read quarters
        
        for r_q in range(2):
            for c_q in range(2):
                for zs in range(chunk_size[0]):
                    cache_id = int(cache_positions[r_q,c_q,zs])
                    if cache_id >=0:
                                                       
                        self.chunk[
                            zs,
                            c_chunk_vox[r_q, c_q, 0, 0]: c_chunk_vox[r_q, c_q, 0, 1]+1,
                            c_chunk_vox[r_q, c_q, 1, 0]: c_chunk_vox[r_q, c_q, 1, 1]+1
                            ] = self.tile_cache.cache[cache_id][ 
                            c_tile_vox[r_q, c_q, 0, 0]:c_tile_vox[r_q, c_q, 0, 1]+1,
                            c_tile_vox[r_q, c_q, 1, 0]:c_tile_vox[r_q, c_q, 1, 1]+1
                            ]
                              
        
    def viewChunk(self, plane = 'None'):
        
        
        chunk_size = self.chunk_list['chunk_size']
        if plane == 'None':
            show_planes = range(chunk_size[0])
        else: 
            show_planes = [plane]
               
        plt.ion()
        fig_num = plt.get_fignums()[-1]
        
        if fig_num:
            fig = plt.figure(fig_num)
        else:
            fig = plt.figure()
            
        # fig_manager = plt.get_current_fig_manager()
        # fig_manager.fig.attributes('-topmost',True)
        # fig.canvas.manager.window('-topmost', True)
        
        fig.clear()
        fig.ax = fig.add_subplot()
        fig.ax.img = fig.ax.imshow(np.zeros((chunk_size[1],chunk_size[2]),np.uint8), cmap='gray', vmin=0, vmax=255)
        fig.show()
                
        for zs in show_planes:
              
            test_image = np.squeeze(self.chunk[zs,:,:].astype(np.uint8))      
            fig.ax.img.set_data(np.squeeze(test_image))

            fig.canvas.flush_events()
            fig.canvas.draw()
            plt.pause(.1)  




if 1: # test
    fd_request = {
        "diced_dir": "//storage1.ris.wustl.edu/jlmorgan/Active/morganLab/DATA/LGN_Developing/KxR_P11LGN/diced/",
        "mip_level": 1,
        "analyze_section": [400], 
        "volume_lower_corner_zyx": [600, 25000, 35000],
        "volume_size_zyx": [32, 1024, 1024],
        "chunk_size_zyx": [16, 256, 256],
        "chunk_overlap": [8, 128, 128]
        }
    
    
    fd = fetchDiced(fd_request)
    
    if 1: # test fetching
        for i in range(fd.chunk_list['num_chunk']):
            
            print(f"fetching chunk {i}")
            fd.readChunk(i)
            fd.viewChunk()
            print('finished fetching chunk')
            #wait = input("press Enter to continue")
    




        
      
    
    
    
    
    



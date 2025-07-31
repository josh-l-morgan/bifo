import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import time
import zarr

from bifo.fetchChunckFromMips import fetchDiced
import bifo.zarrTools as zt

class showInferenceLive:
    
    def __init__(self,v):
                
        self.fig_info = v
        
        # Set up figure
        plt.ion()
                
        fig_num = plt.get_fignums()
        if fig_num:
            self.fig = plt.figure(fig_num[-1])
        else:
            self.fig = plt.figure(figsize=(8,8))
        self.fig.clear()
        self.ax = [self.fig.add_subplot(2,2,d+1) for d in range(4)]
        
        plt.show()
        
        view_shape = v['corner_2'] - v['corner_1']+1

        self.ax[0].img = self.ax[0].imshow(np.zeros(v['clipped_shape'][1:]),cmap='gray', vmin=0, vmax=255)
        self.ax[1].img = self.ax[1].imshow(np.zeros(v['clipped_shape'][1:]),cmap='gray', vmin=0, vmax=255)
        self.ax[2].img = self.ax[2].imshow(np.zeros(view_shape[1:]),cmap='gray', vmin=0, vmax=255)
        self.ax[3].img = self.ax[3].imshow(np.zeros(view_shape[1:]),cmap='gray', vmin=0, vmax=255)

    def update(self, chunk_mid, clipped_mid):
                
        v = self.fig_info
        
        self.ax[0].img.set_data(chunk_mid)
        self.ax[1].img.set_data(clipped_mid)
        zrm_pred_samp = v['zrms'][0][
                    v['center'][0],
                    v['corner_1'][1]: v['corner_2'][1]+1,
                    v['corner_1'][2]: v['corner_2'][2]+1]
        zrm_em_samp = v['zrms'][1][
                    v['corner_1'][1]: v['corner_2'][1]+1,\
                    v['corner_1'][2]: v['corner_2'][2]+1]
        self.ax[3].img.set_data(zrm_pred_samp * 32 + 128)
        self.ax[2].img.set_data(zrm_em_samp )
        
        self.fig.canvas.flush_events()
        self.fig.canvas.draw()
        plt.pause(0.01) 


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

    
def create_zarrs(paths, fd):
    
    z_path = paths['results_dir'] + paths['z_name']
    os.makedirs(paths['results_dir'], exist_ok=True)
    
    z_chunk_shape = fd.valid['clipped_shape']
    zarr_shape = fd.map_request['full_data_shape']    
    
    # Create zarr for whole data set  if necessary
    if 1:#not os.path.isdir(z_path + '/' +  paths['z_group']):
        print(f'creating zarr {z_path}')
        zt.makeMultiscaleGroup(
            z_path = z_path,
            group_name = paths['z_group'],
            zarr_shape = zarr_shape,
            z_chunk_shape = z_chunk_shape,
            num_scales = 9,
            dsamp = [1,2,2]
        )
        
    # Create zarr for tracking progress
    z_pv_path = paths['results_dir'] + 'temp_progress_view.zarr'
    z_pv_group_raw_em = 'raw_em'
    if 1: #not os.path.isdir(z_pv_path + '/' +  z_pv_group_raw_em):
        print(f'creating zarr {z_pv_path}')
        zt.makeMultiscaleGroup(
            z_path = z_pv_path,
            group_name = z_pv_group_raw_em,
            zarr_shape = zarr_shape[1:],
            z_chunk_shape = z_chunk_shape[1:],
            num_scales = 9,
            dsamp = [2,2]
        )
    
def  run_process(paths, map_request, process_request, Network):
    
    fd = fetchDiced(map_request)
    v = fd.valid
    fl = fd.fetch_list
    batch_size = process_request['batch_size']
    
    create_zarrs(paths,fd)
    
    z_path = paths['results_dir'] + paths['z_name']
    z_pv_path = paths['results_dir'] +  'temp_progress_view.zarr'

    os.makedirs(paths['results_dir'], exist_ok=True)
          
    zarr_root = zarr.open_group(z_path, mode='a')
    zrm = zarr_root[paths['z_group']][f'{v['use_mip']}']    
    zarr_pv_root = zarr.open_group(z_pv_path, mode='a')
    zrm_em = zarr_pv_root['raw_em'][f'{v['use_mip']}']
    
        
    # num_fmaps = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    # get model
    model = Network()
    model.load_state_dict(torch.load(paths['cp_path'], weights_only='true')['model_state_dict'])
    model.eval()
    
    # get data from fd to determine positions of prediction assignments
    #furthest_vox = fl["corner_full_vox_2"]
    chunk_shape = v['chunk_shape']
    clp = v['clip']
    clipped_shape = v['clipped_shape']
    c1 = fd.chunk_list["clipped_corners_1"]
    c2 = fd.chunk_list["clipped_corners_2"] 
    fov_center = np.floor((v['fov_corner_2'] + v['fov_corner_2'])/2).astype(int)
    
    # create batch information
    num_batch = int(np.ceil(fd.num_chunk / batch_size))
    batch_array = np.zeros((num_batch, batch_size),'int')-1
    batch_array.flat[0:fd.num_chunk] = np.arange(0,fd.num_chunk)
    
    # Create arrays for assigning prediction chunks
    clipped = np.zeros((clipped_shape),dtype='float32')
    batch = np.zeros((batch_size,1,chunk_shape[0],chunk_shape[1],chunk_shape[2]))
    
    # for display and record
    t = np.zeros((10,2)) # time array
    mid_c = int(np.floor(chunk_shape[0]/2))
    mid_p = int(np.floor(clipped_shape[0]/2))
    c_means = (c1+c2)/2
    diff_to_fov_center = np.abs(c_means - fov_center)   
    chunk_idxs_closest_to_fov_center = np.where(diff_to_fov_center[:,0] == diff_to_fov_center.min(0)[0])[0][0]
    mid_c1 = c1[chunk_idxs_closest_to_fov_center,[0]]
    middle_of_c1 = int(fov_center[0]-mid_c1)
    
    
    # initialize figure shw
    if process_request['show_progress']:
        figi = {
            'zrms': [zrm, zrm_em],
            'corner_1': fl['corner_1'],
            'corner_2': fl['corner_2'],
            'center': fov_center,
            'clipped_shape': v['clipped_shape']
        } 
        shw = showInferenceLive(figi)
    
    for b in range(num_batch):
        
        print(f"processing batch {b} of {num_batch}")
        # read chunks into batch 
        t[0,0] = time.time()
        for c in range(batch_size):    
            fd.readChunk(batch_array[b,c])
            batch[c,0,:] = fd.chunk   
        t[0,1] = time.time()
    
        t[2,0] = time.time()
        input_tensor = torch.tensor(batch).to(device)
        output_tensor = model(input_tensor)
        t[2,1] = time.time()
      
        t[3,0] = time.time()
        for c in range(batch_size):
            cid = batch_array[b,c]
            pred = output_tensor[c].detach().cpu().numpy().squeeze()
            clipped[:].flat = pred[clp[0]:-clp[0], clp[1]:-clp[1], clp[2]:-clp[2]]
            zrm[c1[cid,0]:c2[cid,0]+1,c1[cid,1]:c2[cid,1]+1,c1[cid,2]:c2[cid,2]+1] = clipped
            if c1[cid,0] == mid_c1: # if chunk middle is at center
                zrm_em[c1[cid,1]:c2[cid,1]+1,c1[cid,2]:c2[cid,2]+1] = batch[c, 0, middle_of_c1, clp[1]:-clp[1], clp[2]:-clp[2]]
        t[3,1] = time.time()
        
        # show sample
        if process_request['show_progress']:
            t[4,0] = time.time()    
            shw.update(fd.chunk[mid_c, clp[1]:-clp[1], clp[2]:-clp[2]],
                       clipped[mid_p,:,:] * 20 + 128)
            t[4,1] = time.time()
    
        td = t[:,1] - t[:,0]
        print(f"read {td[0]:.3f}s, expand {td[1]:.3f}s, predict {td[2]:.3f}s, write {td[3]:.3f}s")
    
        
        
if __name__ == '__main__':
    
    # set paths
    paths = {        
        'diced_dir':"F:/KxR_P11_LGN/mips/",
        'cp_path': "G:/Checkpoints/UnetPP_256_01/checkpoints/model_ns12_256by256_rgcDet_round1_itr_10000.pth",
        'results_dir': "G:/Results/pred/",
        'z_name': 'KxR_ms.zarr',
        'z_group': 'pred'
    }
       
    
    map_request_default = {
        "diced_dir": paths['diced_dir'],
        "mip_level": 1,
        "analyze_section": [400], 
        "volume_lower_corner_zyx": [0,0,0],
        "volume_size_zyx": [4, 256, 256], # size of volume at mip level to be processed
        "chunk_shape_zyx": [16, 256, 256], # size to be fed to model
        "chunk_overlap": [8, 128, 128],
        'full_data_shape': [1400, 72000, 50400]
        }
    
    map_request = vast_to_map_request(map_request_default,
                                   vast_location=(15163, 15397, 373),
                                   check_size=[4, 256, 256]
                                   )
    
    process_request = {'batch_size': 1, 'show_progress': 1}   
        
    # run processing
    from bifo.networks.unetplusplus_rgc_short import NestedUNet
    run_process(paths, map_request, process_request, NestedUNet)
    

    

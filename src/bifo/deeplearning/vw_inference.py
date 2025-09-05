# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 19:20:57 2025

@author: jlmorgan

creates vwInfer class
vwInfer runs inverence for multiresolution vw network
Inference batch volumes are assigned according to progress.pkl and plan.pkl

"""

import zarr
import numpy as np
import torch 
import pickle
import time


import bifo.tools.zarr_tools as zt
from bifo.networks.vw_net import threeScaleNestedUNet
import bifo.tools.train_tools as tt
import bifo.tools.vox_tools as vt
import bifo.tools.dtypes as dt

torch.cuda.empty_cache()


class vwInfer():
    """
    Run inference using multi resolution vw network
        
    """
    def __init__(self, paths, infer_request):
        
        self.paths = paths
        self.infer_request = infer_request    
        self.process_request()
        self.initialize_zarrs()
        
   
    def run_plan(self):        
        
        paths = self.paths
        
        reps = 10 ** 10
        for rep in range(reps):
                    
            with open(paths['container'] + paths['plan_path'], 'rb') as f:
                plan = pickle.load(f)
                
            with open(paths['container'] + paths['progress_path'], 'rb') as f:
                progress = pickle.load(f)
                free = (progress['finished'] ==0) & (progress['running']==0)
                pick = np.where(free)[0][0]
                progress['running'][pick] = 1
                progress['start_time'] =  time.time()
            with open(paths['container'] + paths['progress_path'], 'wb') as f:
                pickle.dump(progress, f)
            print()
            self.pick = pick
            print(f'Planning to run batch #{pick} of {len(plan["chkl"])}')
            
            run_chunk = plan['chkl'][pick]
            pw_mip_0 = run_chunk['chunk']['full']
            self.batch_shape = pw_mip_0
                    
            # %% run inference    
            for v in range(3):    
                self.infer_v(pw_mip_0, v)
            
            # %% Blend channels into output    
            self.make_output(pw_mip_0)
                
            # %% Record progress
            with open(paths['container'] + paths['progress_path'], 'rb') as f:
                progress = pickle.load(f)
                progress['running'][pick] = 0
                progress['finished'][pick] = 1
                progress['finished_time'] =  time.time()
            with open(paths['container'] + paths['progress_path'], 'wb') as f:
                pickle.dump(progress, f)         
     
    def process_request(self):
         
        # %% Add to paths
        self.paths['z_group_destination'] = 'temp_inference'
        self.paths['param_path'] = r"param.pkl"
        self.paths['progress_path'] = r"progress.pkl"
        self.paths['plan_path'] = r"plan.pkl"
               
        
        # %% Load and modify train request for inference
        with open(self.paths['container'] + self.paths['param_path'], 'rb') as f:
            tr = pickle.load(f)
        
        tr['v_shapes'] = self.infer_request['v_shapes']
        tr['clip_pred'] = self.infer_request['clip_pred']
        tr['overlap'] = self.infer_request['overlap']
        tr['show_progress_figure'] = self.infer_request['should_show']
        tr['device'] = self.infer_request['device']
        tr['dropout_p'] = 0
        self.interpret_train_request(tr)
        
        # %% Load model
        self.model = threeScaleNestedUNet(tr).to(tr['device'])
    
        # %%        
        ## Buld display
        if tr['show_progress_figure']:
            self.stp = tt.showInferenceProgressV3(tr['v_shapes'][-1][-1])       
    
    
    def initialize_zarrs(self):
        paths = self.paths
            
        # %% initialize zarr files
        self.zr = {}    
        self.zr['root'] = zarr.open_group(paths['z_path'], mode='a')
        self.zr['source'] = self.zr['root'][paths['z_group_source']]
        self.zr['shape0'] = self.zr['source']['0'].shape
        destination_chan = self.tr['input_channels'][-1]
        
        dsamp = self.tr['zarr_dsamp']
    
        if not paths['z_group_destination'] in self.zr['root']:
            new_shape = [destination_chan] + list(self.zr['shape0']) 
            zt.makeMultiscaleGroup(z_path=paths['z_path'], group_name=paths['z_group_destination'], 
                               zarr_shape=new_shape, z_chunk_shape = [destination_chan, 32, 128, 128], 
                               num_scales=9, dsamp=[1] + dsamp)
            
        if not paths['z_group_pred'] in self.zr['root']:
            zt.makeMultiscaleGroup(z_path=paths['z_path'], group_name=paths['z_group_pred'], 
                               zarr_shape=self.zr['shape0'], z_chunk_shape = [32, 128, 128], 
                               num_scales=9, dsamp=dsamp)   
            
        ## Define zarrs        
        self.zr['destination'] = self.zr['root'][paths['z_group_destination']]
        self.zr['prediction'] = self.zr['root'][paths['z_group_pred']]
    
    
    def interpret_train_request(self, tr):
        
        tr['v_vox_sizes'] = []
        for vi, m in enumerate(tr['use_mips']):
            tr['v_vox_sizes'].append(np.array(tr['mip0_vox_size']) 
                                          * np.array(tr['zarr_dsamp']) ** m)
            
            ## crops 
            tr['crop_v'] = []
            for k in range(len(tr['use_mips'])):
                tr['crop_v'].append([0, 0, tr['crop_loss'][k][0], tr['crop_loss'][k][1], tr['crop_loss'][k][2]])
    
        self.tr = tr
    
    
    def infer_v(self, pw_mip_0, v):
                
        paths = self.paths
        tr = self.tr
        
        zrm = self.zr['source'][f"{tr['use_mips'][v]}"]
        zr_dest = self.zr['destination'][f"{tr['use_mips'][v]}"]
        
        save_chan_num = tr['input_channels'][-1] - 1
        
        torch.cuda.empty_cache()
        
        ## Load mode
        #model = threeScaleNestedUNet(tr).to(tr['device'])
        checkpoint_path = paths['container'] + paths['checkpoint_path']
        checkpoint = torch.load(checkpoint_path, weights_only='true')
        self.model.load_state_dict(checkpoint['model_state_dict'])
           
        ## Get chunk parameters
        pws = []
        dsamp = np.array(tr['zarr_dsamp'] )
        for ri in range(len(np.array(tr['v_vox_sizes']))):
            pws.append((pw_mip_0 / dsamp ** tr['use_mips'][ri]).astype(int))
            
        overlap = np.array(tr['overlap'][v])
        #valid_shape = np.array(tr['v_shapes'][v]) - clip_p * 2
        
        pw = pws[v].astype(int)
        aw = np.array(([0, 0, 0], zrm.shape[-3:]))
        chkl = vt.list_chunks_overlap(pw, tr['v_shapes'][v], overlap=overlap, aw=aw)
        num_chunks = len(chkl)
        
        if v > 0:
            zr_dest_previous = self.zr['destination'][f"{tr['use_mips'][v-1]}"]
        
        ## initialize run variables
        z_in = [''] * 3
        input_tensor = [''] * 3
        tk = dt.ticTocDic()
        tk.m(['run','itt','pull','pred','write','show'])
        
        ## Make blending mask
        valid = vt.make_blend_mask(
            tile_shape=np.array(tr['v_shapes'][v]),
            clip_vec=np.array(tr['clip_pred'][v]),
            overlap_vec=np.array(tr['overlap'][v]),
            mode="cosine"  # "cosine" or "linear"
        )
            
        ## Run
        tk.b('run')
        for ci in range(num_chunks):
            tk.b('itt')
            
            tk.b('pull')
            ## Get raw input from appropritate mip level
            ch = chkl[ci]
            # zw_ch = ch['chunk']['ch']
            # zw_in = ch['chunk']['full']
            # zw_ch = ch['chunk_complete']['ch']
            # zw_in = ch['chunk_complete']['full']
            
            chunk_request= ch['chunk_complete']['full']
            chunk_aw = vt.block_to_window(aw,chunk_request)                 
            zw_ch = chunk_aw['ch']
            zw_in = chunk_aw['full']
            zw_out =  np.concatenate((np.array([[0],[save_chan_num]]), zw_in),1)
            z_in = vt.get_win(zrm, zw_in)    
            input_tensor = [torch.from_numpy(z_in).unsqueeze(0).unsqueeze(0)]
        
            ## pull previous v, if not the first v
            if v > 0: 
                input_shape = zw_in[1,:]-zw_in[0,:]
                same = tuple([slice(0,int(s)) for s in input_shape])
                same2 = (slice(0, save_chan_num, None),) + same
                zw_previous = vt.downsample_win(zw_in, dsamp)
                previous_pred = vt.get_win(zr_dest_previous, zw_previous)
                previous_scaled = previous_pred[0:-1,:,:,:]
                previous_upsampled = vt.upsample_array(previous_scaled, dsamp)
                previous_tensor = torch.from_numpy(previous_upsampled[same2]).unsqueeze(0)
                input_tensor = [torch.cat([input_tensor[0], previous_tensor], dim=1)]
            else: 
                previous_upsampled = None
            tk.e('pull')
               
            tk.b('pred')  
            min_shape = np.array(input_tensor[0].shape[2:]).min()
            if min_shape < 3:
                print('input too small for network')
            else:
                preds, feats = self.model(input_tensor, v)
                tk.e('pred')
            
                tk.b('write')
                ## sort based on v stage
                if isinstance(preds, list):
                    num_pred = len(preds)
                    pred_shape = preds[0].shape
                    save_pred = np.zeros([save_chan_num, pred_shape[2],pred_shape[3], pred_shape[4]])
                    for ti, t in enumerate(preds):
                        save_pred[ti,:,:,:] = t.detach().numpy()
                    show_pred = save_pred[num_pred-1,:,:,:]
                else:
                    save_pred = feats[0,0:save_chan_num,:,:,:].detach().numpy()
                    show_pred = preds[0,0,:,:,:].detach().numpy()
               
                ## Write to zarr
                zw_valid = zw_out.copy()
                zw_valid[:,0] = np.array([save_chan_num,save_chan_num+1],int)
                
                prev_valid = vt.get_win(zr_dest, zw_valid) 
                prev_pred = vt.get_win(zr_dest, zw_out)
                #cut_weight = vt.get_win(bp.cvv.masks[v], zw_ch)
                cut_weight = vt.get_win(valid, zw_ch)
                
                new_valid = prev_valid + cut_weight
                pred_scaled = save_pred * cut_weight # reduce saved values according to validity
                new_pred = prev_pred + pred_scaled     
                
                vt.put_win(zr_dest, zw_valid, new_valid)
                vt.put_win(zr_dest, zw_out, new_pred)  
                tk.e('write')
                   
                # show progress, add 'all' to see movide of all planes
                tk.b('show')
                if tr['show_progress_figure']:
                    self.stp.update(input_t=z_in, pred_t=show_pred,
                               prev=save_pred)
                tk.e('show')
                    
                print(f'pick {self.pick}, v {v}, chunk {ci} of {num_chunks}. position {zw_in[0,:]}')
                tk.e('itt')
                tk.pl()         
    
   
            
    def make_output(self, pw_mip_0):
            # %% make blended zarr
            dsamp = np.array(self.tr['zarr_dsamp'])
            use_channel = 2
            
            write_mip = self.tr['use_mips'][-1]
            zrm_p = self.zr['prediction'][f'{write_mip}']
            zrm_s = self.zr['destination'][f'{write_mip}']
            
            
            pw = (pw_mip_0 / dsamp ** write_mip).astype(int)
            aw = np.array([[0, 0, 0], zrm_p.shape])
            chkl = vt.list_chunks_overlap(pw, zrm_p.chunks, overlap=None, aw=aw)
            num_chunk = len(chkl)
            
            blend_channel = zrm_s.shape[0]-1
            for ci in range(num_chunk):
                print(f'writing chunk {ci} of {num_chunk}')
                cw = chkl[ci]['chunk_complete']['full']
                slice_cw = vt.win_to_slices(cw)
                slice_blend = (blend_channel,) + slice_cw
                slice_use = (use_channel,) + slice_cw
                vol_b = zrm_s[slice_blend]
                vol_p = zrm_s[slice_use]
                vol_s = np.divide(vol_p, vol_b, 
                                 out=np.zeros_like(vol_p, dtype=vol_p.dtype),
                                 where=(vol_b > 0))
                zrm_p[slice_cw] = vol_s
            
            
    def scan_result(self, pw_mip_0=None):                
        
            if pw_mip_0 is None:
                pw_mip_0 = self.batch_shape
                
            pw_show = pw_mip_0.copy()
            middle = pw_show[:,0].mean().astype(int) 
            pw_show[:,0] = np.array([middle,middle+1])
            
        
            zt.scan_zarr_to_fig(zarr_group=[ self.zr['source'], self.zr['destination'], self.zr['prediction']],
                             pw=pw_show, 
                             cw=[1,1024,1024], 
                             scan_mips=[1], 
                             dsamp=[1, 2, 2],
                             use_channel=[0, 4, 0])
        
           
            # #check that downsample went well    
            # zt.scan_zarr_to_fig(zarr_group=[self.zr['source'], self.zr['destination'], self.zr['prediction'],],
            #                  pw=pw_show, 
            #                  cw=[1,1024,1024], 
            #                  scan_mips=[1, 2, 3], 
            #                  dsamp=[1, 2, 2],
            #                  use_channel=[0, 4, 0])






import torch.utils.data as data
import torch
import h5py
import numpy as np
import random
import cv2
from scipy import misc
from math import ceil
import random

class DatasetFromHdf5(data.Dataset):
    def __init__(self, opt):
        super(DatasetFromHdf5, self).__init__()
        
        hf = h5py.File(opt.dataset_path)
        self.LFI = hf.get('LFI')  # [N,ah,aw,h,w] 4x7x7x512x512
        self.LFI = self.LFI[:, :opt.angular_out, :opt.angular_out, :, :]
   
        self.psize = opt.patch_size
        self.ang_out = opt.angular_out
        self.ang_in = opt.angular_in

    
    def __getitem__(self, index):
                        
        # get one item
        lfi = self.LFI[index]  # [ah,aw,h,w]

        # crop to patch
        H = lfi.shape[2]
        W = lfi.shape[3]

        # print('H: ', H)
        # print('W: ', W)
        x = random.randrange(0, H-self.psize)
        y = random.randrange(0, W-self.psize) 
        lfi = lfi[:, :, x:x+self.psize, y:y+self.psize]
        
        # 4D augmentation
        # flip
        if np.random.rand(1)>0.5:
            lfi = np.flip(np.flip(lfi,0),2)
        if np.random.rand(1)>0.5:
            lfi = np.flip(np.flip(lfi,1),3)
        # rotate
        r_ang = np.random.randint(1,5)
        lfi = np.rot90(lfi,r_ang,(2,3))
        lfi = np.rot90(lfi,r_ang,(0,1))
            
        
        ##### get input index ######
        ind_all = np.arange(self.ang_out*self.ang_out).reshape(self.ang_out, self.ang_out)
        delt = (self.ang_out-1) // (self.ang_in-1)
        ind_source = ind_all[0:self.ang_out:delt, 0:self.ang_out:delt]
        ind_source = ind_source.reshape(-1)

        ##### get input and label ######    
        lfi = lfi.reshape(-1, self.psize, self.psize)
        input = lfi[ind_source, :, :]

        NumView2 = 2
        a = 0
        LF2 = np.zeros((1, self.psize * NumView2, self.psize* NumView2))
        for i in range(NumView2):
            for j in range(NumView2):
                img = input[a,:,:]
                img= img[np.newaxis,:,:]
                LF2[:, i::NumView2, j::NumView2] = img
                a = a +1
        lenslet_data = LF2

        H = self.psize
        W = self.psize
        allah = self.ang_in
        allaw = self.ang_in
        LFI = np.zeros((1, H, W, allah, allaw))
        eslf = lenslet_data
        for v in range(allah):
            for u in range(allah):
               sub = eslf[:, v::allah, u::allah]
               LFI[:,:,:, v, u] = sub[:, 0:H, 0:W]
        LFI = LFI.reshape(self.psize, self.psize, 2, 2)

        # to tensor
        input = torch.from_numpy(input.astype(np.float32)/255.0)
        label = torch.from_numpy(lfi.astype(np.float32)/255.0)
        LFI = torch.from_numpy(LFI.astype(np.float32)/255.0)

        return ind_source, input, label, LFI

            
    def __len__(self):
        return self.LFI.shape[0]



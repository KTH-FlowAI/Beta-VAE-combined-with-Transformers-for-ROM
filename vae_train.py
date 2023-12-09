"""
Training Script for beta-VAE model using fluctuation and VAE using Tanh as activation function
@yuningw
"""
# Basic Env
import os
import time 
import torch
import torch.nn as nn 
import h5py
import numpy as np 

# Arch 
from utils.VAE.AutoEncoder import BetaVAE

# Utils
from utils.vae_training_utils import fit
from utils.datas import make_DataLoader
from utils.configs import VAE_custom as args, Name_Costum_VAE
# Assign the random seed
torch.manual_seed(1024)
# Confirm the device
device = ("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Confirm the device {device}")

os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]  = "0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Unlock the h5py file


base_dir        = os.getcwd() 
base_dir        += "/"
print(f"Current dir : {base_dir}")
datafile        = '01_Data/u_1_to_26.hdf5'
check_point_dir = "02_Checkpoints/"

with h5py.File(base_dir + datafile, 'r') as f:
  u_keras   = np.array(f['u'][:],dtype=np.float32)
  nt,nx,ny  = f['nt'][()], f['nx'][()],f['nz'][()]
  u_mean    = f['mean'][:]
  u_std     = f['std'][:]
f.close()
print(f"The dataset has been loaded, the hdf5 file closed")

u_keras     = np.nan_to_num(u_keras)

if args.act_conv == "tanh":  
  u_keras     = u_keras * u_std
  print(f"Use {args.act_conv} as activation, so we use fluctuation data")


u_keras     = np.transpose( u_keras, (0,2,1))
u_keras     = u_keras[:,np.newaxis,:,:]

# Get the training data
Ntrain      = int(args.test_split* nt)
u_keras     = u_keras[:Ntrain]
print(f"INFO:Pre-processing ended, \n"+\
      f"whole dataset has shape of {u_keras.shape},\n"+\
      f"We use first {Ntrain} for the training data  ")


t_dl, v_dl = make_DataLoader(X            = torch.from_numpy(u_keras),
                             y            = torch.from_numpy(u_keras),
                             train_split  = args.train_split ,
                             batch_size   = args.batch_size)
print(f"INFO: The data has been splited by ratio of train:val = {args.train_split}")

fileID = Name_Costum_VAE(args, nt)

print(f"INFO: The fileID: {fileID}")

# Model type: v2-Arch1, v3-Arch2
model = BetaVAE(    zdim         = args.latent_dim, 
                    knsize       = args.knsize, 
                    beta         = args.beta, 
                    filters      = args.filters,
                    block_type   = args.block_type,
                    lineardim    = args.linear_dim,
                    act_conv     = args.act_conv,
                    act_linear   = args.act_linear)
print(f"INFO: Architecture has been bulit")

model.to(device)

opt = torch.optim.Adam(model.parameters(), lr =args.lr, eps=1e-7,weight_decay=args.w_decay)

# The setting for LR scheduler
NumDivide     = 5 
milestones    = np.linspace(args.epoch//NumDivide,args.epoch,NumDivide)
decay_ratio   = 0.8

opt_scheduler = [
                  torch.optim.lr_scheduler.MultiStepLR(optimizer=opt, 
                                                      milestones=milestones,
                                                      gamma=decay_ratio) 
                ]
print(f"INFO: We use multi-step LR decay,\nIt will {decay_ratio} * LR at {milestones} ")

print(f"Model and optimizer have been correctly compiled")

start_time  = time.time()
history     = fit(args.epoch, model, 
                opt, opt_scheduler,
                t_dl,v_dl,device,
                earlystop=args.earlystop, patience=args.patience)
end_time    = time.time()
cost_time   = end_time - start_time
print(f"INFO: Training Ended, time = {np.round(cost_time,4)}s")


check_point = {"model":model.state_dict(),
               "history":history,
               "time":cost_time,
               }
torch.save(check_point,
           base_dir + check_point_dir +fileID+".pt")
print(f"INFO: The checkpoints has been saved")

"""
Rank the non-linear modes for VAE
@author yuningw
"""

import os
from utils.VAE.AutoEncoder import BetaVAE
from utils.configs import VAE_custom as args, Name_Costum_VAE
import torch
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np 
from tqdm import tqdm
import time
from utils.plot import colorplate as cc 
from utils.pp import *

torch.manual_seed(1024)
device = ("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Unlock the h5py file


batch_size = args.batch_size
beta = args.beta
latent_dim = args.latent_dim
epochs = args.epoch
split_ratio = args.train_split
lr = args.lr
es =  args.earlystop
model_type = args.model

CheckPoint_path = "02_Checkpoints/"
datafile        = '01_Data/u_1_to_26.hdf5'
with h5py.File( datafile, 'r') as f:
  u_keras = f['u'][:]
  nt = int(f['nt'][()])
  nx = int(f['nx'][()])
  ny = int(f['nz'][()])
  xx = np.array(f['xx'])
  zz = np.array(f['zz'])
  u_mean = f['mean'][:]
  u_std = f['std'][:]

u_keras = np.nan_to_num(u_keras)
u_keras = np.transpose( u_keras, (0,2,1))
u_keras = u_keras[:,np.newaxis,:,:]

print(f"The shape of data: {u_keras.shape}")

# We treat VAE as a method of modal decomposition,so we use whole dataset for test
u_t     = torch.tensor(u_keras)
u       = TensorDataset(u_t, u_t)
dl      = DataLoader(u ,batch_size = 1)

fileID  = Name_Costum_VAE(args, nt)
print(f"The fileID will be {fileID}")

ckpt    = torch.load(CheckPoint_path +fileID+".pt",map_location=device)
model    = BetaVAE(    
                    zdim         = args.latent_dim, 
                    knsize       = args.knsize, 
                    beta         = args.beta, 
                    filters      = args.filters,
                    block_type   = args.block_type,
                    lineardim    = args.linear_dim,
                    act_conv     = args.act_conv,
                    act_linear   = args.act_linear)

model.load_state_dict(ckpt["model"])
model.to(device)
model.eval()
print("INFO: Model has been correctly loaded")

######
#Temporal Modes
######
print(f"INFO: Generating temporal modes")

Z_vector = [];  Z_mean = []; Z_var = []
for x,y in tqdm(dl):
    x                   = x.float().to(device)
    z_mean, z_var, pred = model(x)
    z_sample            = model.reparameterize((z_mean, z_var))

    Z_vector.append(z_sample.detach().cpu().numpy())
    Z_mean.append(z_mean.detach().cpu().numpy())
    Z_var.append(z_var.detach().cpu().numpy())

Z_vector    = np.array(Z_vector).squeeze()
Z_mean      = np.array(Z_mean).squeeze()
Z_var       = np.array(Z_var).squeeze()

print(f"INFO: Modes generated, the shape of vectors are {Z_vector.shape}")
del dl, u, u_t, z_mean, z_var
print(f"INFO: Deleted the tensor Dataset and Dataloader")
s_t         = time.time()
ranks, Ecum = Rank_SpatialMode( model        = model, 
                                latent_dim   = args.latent_dim, 
                                u_truth      = u_keras,
                                u_std        = u_std, 
                                modes        = Z_vector,
                                device       = device)

e_t         = time.time()

print(f"INFO: Rank Finished, the time cost = {np.round(e_t - s_t)}s")

SpatialModes    =  Gen_SpatialMode(model        = model, 
                                   latent_dim   = args.latent_dim,
                                   device       = device,
                                   invalue      = 1)

print(f"The type of data:\n"  +f"Spatial Modes = {type(SpatialModes)}" + f"z mean =  {type(Z_mean)}\n" + f"z var  =  {type(Z_var)}\n" + f"rank   =  {type(ranks)}"
      )

np.savez_compressed(
                        "03_Mode/"+ "Rank_Mode_" + fileID + ".npz",
                        modes       = SpatialModes,
                        z_mean      = Z_mean, 
                        z_var       = Z_var,
                        ranks       = ranks,

                        )

print(f"Temporal and Spatial Modes are saved")

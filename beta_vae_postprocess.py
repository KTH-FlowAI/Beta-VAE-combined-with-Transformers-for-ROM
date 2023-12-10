"""
Evaluate the performance of beta-VAE
"""
import  os
import  h5py
import  numpy as np 
import  matplotlib.pyplot as plt 
from    utils.VAE.AutoEncoder import BetaVAE
from    utils.configs         import VAE_custom as args, Name_Costum_VAE
import  torch
from    torch.utils.data      import DataLoader, TensorDataset
from    tqdm                  import tqdm
from utils.plot               import colorplate as cc 
from utils.pp                 import Energy_Rec, Gen_SpatialMode, Corr_Martix

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

CheckPoint_path = "02_CheckPoints/"
datafile        = '01_Data/u_1_to_26.hdf5'
csv_file        = "vae_results.csv"

with h5py.File(datafile, 'r') as f:
  u_keras = f['u'][:]
  nt = int(f['nt'][()])
  nx = int(f['nx'][()])
  ny = int(f['nz'][()])
  u_mean = f['mean'][:]
  u_std = f['std'][:]

u_keras = np.nan_to_num(u_keras)
u_keras = np.transpose( u_keras, (0,2,1))
u_keras = u_keras[:,np.newaxis,:,:]

print(f"The shape of data: {u_keras.shape}")

Ntrain      = int(args.test_split* nt)

# We treat VAE as a method of modal decomposition,so we use whole dataset for test
u_t     = torch.tensor(u_keras)
u       = TensorDataset(u_t, u_t)
dl      = DataLoader(u ,batch_size = 1)

fileID  = Name_Costum_VAE(args, 26000)
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

Z_mean = []; Z_var = []; Pred = []; Z_vector = []
for x,y in tqdm(dl):
    x   = x.float().to(device)
    z_mean, z_var, pred = model(x)
    z_sample = model.reparameterize((z_mean, z_var))

    Z_mean.append(z_mean.detach().cpu().numpy())
    Z_var.append(z_var.detach().cpu().numpy())
    Z_vector.append(z_sample.detach().cpu().numpy())
    Pred.append(pred.detach().cpu().numpy())

Z_mean  = np.array(Z_mean).squeeze()
Z_var   = np.array(Z_var).squeeze()
Z_vector   = np.array(Z_vector).squeeze()
Pred    = np.array(Pred).squeeze()

print(f"The shape of vectors are {Z_vector.shape}")


print(f"Computing Det_R")
det_id      = 'det_'+ fileID
det_path    = "04_Figs/"+det_id


detR, corr_matrix_latent = Corr_Martix(z_mean=Z_mean)

fig, ax     = plt.subplots(figsize=(10, 10))
im          = ax.imshow(corr_matrix_latent, cmap='rainbow')
ax.grid(False)
for i in range(latent_dim):
    for j in range(latent_dim):
        ax.text(j, i, round(corr_matrix_latent[i, j],2), ha='center', va='center',
                color='k')
cbar        = ax.figure.colorbar(im, ax=ax, format='% .2f')
plt.title(f"Det = {np.round(detR,4)},beta={beta},modes={latent_dim}",{"size":18})
plt.savefig(det_path,dpi=200,bbox_inches="tight")
print(f"The determination of correlation matrix is {np.round(detR,4)}")




Modes = Gen_SpatialMode(latent_dim  =   args.latent_dim, 
                        model       =   model, 
                        device      =   device,
                        invalue     =   1)
print("INFO: Spatial Modes has been generated")


det_id      = 'Mode_'   + fileID
det_path    = "04_Figs/"+det_id
fig,axs     = plt.subplots(latent_dim,1,figsize= (5.5, latent_dim*1.2))
for i in range(latent_dim):
    axs[i].imshow(Modes[i,:,:],cmap= "RdBu")
    axs[i].axis("off")
plt.savefig(det_path,dpi = 150, bbox_inches = 'tight')
print("INFO: Spatial Modes have been plotted")


np.savez_compressed(    "03_Mode/"+fileID+"modes.npz",
                        z_mean      = Z_mean,
                        z_var       = Z_var,
                        modes       = Modes,
                        corr_matrix = corr_matrix_latent, 
                        )

print(f"INFO: Temporal and Spatial Modes are saved")



print("INFO: Computing reconstruction energy")
u_p         = Pred*u_std
u           = u_keras.squeeze()*u_std
num_fields  = u.shape[0]


Energy_rec  =Energy_Rec(truth= u, pred= u_p)

print(f"Reconstruction Energy={np.round(Energy_rec,2)}%")


snap_id     = 'Snap_'   + fileID
snap_path   = "04_Figs/"+snap_id

u_plot = np.concatenate([
                            u[num_fields//2:num_fields//2+1,:,:],
                            u_p[num_fields//2:num_fields//2+1,:,:]
                        ],axis=0)
print(u_plot.shape)

ny, nx  = u_plot.shape[1],u_plot.shape[2]
x       = np.linspace(-1, 5, nx)
y       = np.linspace(-1.5, 1.5, ny)
y, x    = np.meshgrid(y, x)
x       = x[:192, :96]
y       = y[:192, :96]
xb      = np.array([-0.125, -0.125, 0.125, 0.125, -0.125])
yb      = np.array([-0.125, 0.125,  0.125, -0.125, -0.125])

names   = ["Ground Truth",f"Reconstruction Energy={np.round(Energy_rec,2)}%"]

fig,ax  = plt.subplots(2, sharex = True, sharey = True, figsize = (16, 10))
plt.set_cmap('YlGnBu_r')
for i in range(len(ax)):
    cb=ax[i].contourf(x,y,u_plot[i,:,:].T,levels=300, vmin = u.min(),vmax = u.max())
    ax[i].fill(xb, yb, c = 'w',zorder =3)
    ax[i].plot(xb, yb, c = 'k', lw = 1, zorder = 5)
    ax[i].set_aspect('equal')
    ax[i].set_title(names[i],fontdict={"size":18})
    plt.colorbar(cb,ax= ax[i])
plt.tight_layout()
plt.savefig(snap_path,dpi=300, bbox_inches="tight")

print(f"INFO: Reconstruction Figure Saved")
print(f"INFO: Plotting Loss VS Epochs")

print("#"*20)
snap_id     = 'Loss_'   + fileID
snap_path   =  "04_Figs/"+snap_id
history     = ckpt["history"]

fig, axs = plt.subplots(1,1, figsize =(8,4))

axs.semilogy(history["rec_loss"],"-",lw = 2, c = cc.red,label="rec loss")
axs.semilogy(history["val_rec_loss"],"--",lw = 2, c = cc.red,label="val rec loss")
ax2 = axs.twinx()
ax2.plot(history["kl_loss"],"-",lw =2, c = cc.blue,label = "val_kl_loss" )
ax2.plot(history["val_kl_loss"],"--",lw =2, c = cc.blue,label = "val_kl_loss" )

axs.tick_params("y",colors = cc.red)
axs.yaxis.label.set_color(cc.red)
ax2.tick_params("y",colors = cc.blue)
ax2.yaxis.label.set_color(cc.blue)
ax2.spines["left"].set_color(cc.red)
ax2.spines["right"].set_color(cc.blue)

axs.set_xlabel("Epoch",fontsize = 18)
axs.set_ylabel("Rec Loss",{"color":cc.red},fontsize = 18)
ax2.set_ylabel("KL Divergence",{"color":cc.blue},fontsize = 18)
plt.savefig(snap_path,dpi = 150, bbox_inches="tight")

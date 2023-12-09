"""
Train easy attention on the mu only 
"""

import  torch
from    utils.NNs.easyAttns import  easyTransformerEncoder
import  os
import  h5py
import  time
import  numpy               as      np
from    matplotlib          import  pyplot  as  plt
from    utils.configs       import  EasyAttn_config as cfg, VAE_custom, Name_Costum_VAE, Make_Transformer_Name
from    utils.pp            import make_Prediction
from    utils.plot          import plot_signal, colorplate as cc 
from    utils.VAE.AutoEncoder import BetaVAE
import  argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model","-m", default="self", type=str)
args  = parser.parse_args()

torch.manual_seed(1024)
np.random.seed(1024)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
#confirm device
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


vae_name                =   Name_Costum_VAE(VAE_custom,nt= 25999)
base_dir                =   os.getcwd()
base_dir                +=  "/"
datafile                = base_dir +  '01_Data/u_1_to_26.hdf5'
checkpoint_save_path    = base_dir +  "06_ROM/OnlyPredictor/CheckPoints/"
CheckPoint_path         = base_dir +  "02_Checkpoints/"
modes_data_path         = base_dir +  "03_Mode/"
save_fig_pred           = base_dir +  f"04_Figs/vis_pred/dim{VAE_custom.latent_dim}/"
save_window_pred        = base_dir +  f"04_Figs/vis_pred/sliding_window/"
save_snap_pred          = base_dir +  f"04_Figs/vis_pred/snapshots/"
save_data_pred          = base_dir +  f"06_Preds/dim{VAE_custom.latent_dim}/"
save_fig_chaotic        = base_dir +  f"04_Figs/vis_pred/chaotic/"


if args.model == "easy":
  from    utils.configs       import  EasyAttn_config as cfg, VAE_custom, Name_Costum_VAE, Make_Transformer_Name
  from    utils.NNs.easyAttns import  easyTransformerEncoder

  print("#"*30)
  fileID                  =   Make_Transformer_Name(cfg)
  vae_name                =   Name_Costum_VAE(VAE_custom,nt= 25999)
  fileID                  =   "Mean_" + fileID + "_" + vae_name
  print(f"INFO: the fileID is\n{fileID}")
  print("Loading model")
  preditor = easyTransformerEncoder(
                                      d_input = cfg.in_dim,
                                      d_output= cfg.next_step,
                                      seqLen  = cfg.nmode,
                                      d_proj  = cfg.time_proj,
                                      d_model = cfg.d_model,
                                      d_ff    = cfg.proj_dim,
                                      num_head = cfg.num_head,
                                      num_layer = cfg.num_block,
                                      )

  NumPara = sum(p.numel() for p in preditor.parameters() if p.requires_grad)
  print(f"INFO: The model has been generated, num of parameter is {NumPara}")

  stat_dict   =   torch.load(checkpoint_save_path+fileID+".pt", map_location= device)

  preditor.load_state_dict(stat_dict['model'])
  print("INFO: The weight of predictor has been loaded")

elif args.model == "self":
  from    utils.configs       import  Transformer_config as cfg, VAE_custom, Name_Costum_VAE, Make_Transformer_Name
  from    utils.NNs.EmbedTransformerEncoder import  EmbedTransformerEncoder
  print("#"*30)
  fileID                  =   Make_Transformer_Name(cfg)
  vae_name                =   Name_Costum_VAE(VAE_custom,nt= 25999)
  fileID                  =   "Mean_" + fileID + "_" + vae_name
  preditor   = EmbedTransformerEncoder(d_input = cfg.in_dim,
                                    d_output= cfg.next_step,
                                    n_mode  = cfg.nmode,
                                    d_proj  = cfg.time_proj,
                                    d_model = cfg.d_model,
                                    d_ff    = cfg.proj_dim,
                                    num_head = cfg.num_head,
                                    num_layer = cfg.num_block,
                                    )
  

  NumPara = sum(p.numel() for p in preditor.parameters() if p.requires_grad)
  print(f"INFO: The model has been generated, num of parameter is {NumPara}")

  stat_dict   =   torch.load(checkpoint_save_path+fileID+".pt", map_location= device)

  preditor.load_state_dict(stat_dict['model'])
  print("INFO: The weight of predictor has been loaded")

elif args.model =='lstm':
  from    utils.configs       import  LSTM_config as cfg, VAE_custom, Name_Costum_VAE, Make_LSTM_Name
  from    utils.NNs.RNNs      import  LSTMs
  print("#"*30)
  fileID                  =   Make_LSTM_Name(cfg)
  vae_name                =   Name_Costum_VAE(VAE_custom,nt= 25999)
  fileID                  =   "Mean_" + fileID + "_" + vae_name
  preditor    = LSTMs(
                d_input= cfg.in_dim, d_model= cfg.d_model, nmode= cfg.nmode,
                embed= cfg.embed, hidden_size= cfg.hidden_size, num_layer= cfg.num_layer,
                is_output= cfg.is_output, out_dim= cfg.next_step, out_act= cfg.out_act
                )
  NumPara = sum(p.numel() for p in preditor.parameters() if p.requires_grad)
  print(f"INFO: The model has been generated, num of parameter is {NumPara}")

  stat_dict   =   torch.load(checkpoint_save_path+fileID+".pt", map_location= device)

  preditor.load_state_dict(stat_dict['model'])
  print("INFO: The weight of predictor has been loaded")
else:
  print("ERROR: There is no type match!")
  quit()

print("#"*30)
print(f"Loading data")
with h5py.File(datafile, 'r') as f:
  u_keras   = np.array(f['u'][:],dtype=np.float32)
  nt,nx,ny  = f['nt'][()], f['nx'][()],f['nz'][()]
  u_mean    = f['mean'][:]
  u_std     = np.array(f['std'][:])
f.close()
print(f"The dataset has been loaded, the hdf5 file closed")

u_keras     = np.nan_to_num(u_keras)
u_keras     = np.transpose( u_keras, (0,2,1))
u_keras     = u_keras[:,np.newaxis,:,:]

Ntrain      =   int(cfg.train_split* nt)
u_test      =   u_keras[Ntrain: ,:,:]
print(f"Load the snapshots with shape = {u_test.shape}")

d           =   np.load(modes_data_path + vae_name + "modes.npz")
z_mean      =   d["z_mean"]
z_var       =   d["z_var"]

Nt          =   z_mean.shape[0]
Ntrain      =   int(cfg.train_split* Nt)
# Ntrain      =   -2000
print(z_mean.shape)
# quit()
train_mean  =   z_mean[:Ntrain]
test_mean   =   z_mean[Ntrain:]
print(f"INFO: Load latent variable: Ntrain = {Ntrain}. train mean = {train_mean.shape}, test_mean = {test_mean.shape}")
train_var   =   z_var[:Ntrain,:]
test_var    =   z_var[Ntrain:,:]
test_data   =   test_mean
print("#"*30 )
print("The prediction of snapshot")

fileVAE  = Name_Costum_VAE(VAE_custom, 25999)
print(f"The fileID will be {fileID}")
ckpt     = torch.load(CheckPoint_path  +fileVAE+".pt",map_location=device)
model    = BetaVAE(    
                    zdim         = VAE_custom.latent_dim, 
                    knsize       = VAE_custom.knsize, 
                    beta         = VAE_custom.beta, 
                    filters      = VAE_custom.filters,
                    block_type   = VAE_custom.block_type,
                    lineardim    = VAE_custom.linear_dim,
                    act_conv     = VAE_custom.act_conv,
                    act_linear   = VAE_custom.act_linear)

model.load_state_dict(ckpt["model"])
model.to(device)
model.eval()
print("INFO: Model has been correctly loaded")
pred_mean       = make_Prediction(test_data, preditor,device,
                        in_dim= cfg.in_dim, 
                        next_step= cfg.next_step)

plot_signal(test_data=test_mean, Preds=pred_mean,save_file=save_fig_pred + "Singal_"+ fileID +".jpg")


Physical_prediction   =   []
z_sample              =   []

for i in range(pred_mean.shape[0]):
    z_mean_   =   torch.from_numpy(pred_mean[None,i,np.newaxis,:]).float().to(device)
    z_var_    =   torch.from_numpy(test_var[None,i,np.newaxis,:]).float().to(device)
    z_sample_ = model.reparameterize((z_mean_, z_var_))
    snap      =   model.decoder(z_sample_)
    Physical_prediction.append(snap.detach().cpu().numpy())
    z_sample.append(z_sample_.detach().cpu().numpy())

Physical_prediction = np.concatenate(Physical_prediction,0)


np.savez_compressed(save_data_pred +"Pred_Data_" + fileID + ".npz", 
                    
                    p_zmean =  pred_mean,
                    p_snap  = Physical_prediction,
                    Ntrain  = Ntrain
                    )

###
## Check the snapshot 
####

Nx, Ny  =   192, 96 
x       =   np.linspace(-1, 5, Nx)
y       =   np.linspace(-1.5, 1.5, Ny)
y, x    =   np.meshgrid(y, x)
x       =   x[:192, :96]
y       =   y[:192, :96]
xb      =   np.array([-0.125, -0.125, 0.125, 0.125, -0.125])
yb      =   np.array([-0.125, 0.125, 0.125, -0.125, -0.125])


N_tot   =   Physical_prediction.shape[0]

No_eval =   [1 , 51, 76, 101]

print(Physical_prediction.shape)
print(u_test.shape)
Physical_prediction *= u_std
u_test              *= u_std

fig, ax = plt.subplots(len(No_eval),2, figsize = (8,10), sharex= True, sharey=True)
for i, no_eval in enumerate(No_eval):
  # Note that we use the prediction which is at (T+1) step
  up      =   Physical_prediction[128 + no_eval,0,:,:].T 
  ug      =   u_test[128 + no_eval,0,:,:].T 
  E_k     =   1 - np.mean((up - ug )**2)/np.mean(ug **2 )
  E_k     *=  100 
  
  u_all   =   np.concatenate([up,ug])

  clb1    = ax[i, 0].contourf(x,y, up, 
                           cmap= "YlGnBu_r", 
                           levels= 100, 
                           vmin = u_all.min(),
                           vmax = u_all.max()
                           )

  clb2    = ax[i, 1].contourf(x,y, ug, 
                            cmap= "YlGnBu_r", 
                            levels= 100,
                            vmin = u_all.min(),
                            vmax = u_all.max()
                              
                            )

  ax[i, 0].set_aspect("equal")
  ax[i, 1].set_aspect("equal")
  ax[-1, 0].set_xlabel("x")
  ax[-1, 1].set_xlabel("x")
  ax[i, 0].set_ylabel("z")
  ax[i, 0].fill(xb, yb, c = 'w',zorder =3)
  ax[i, 1].fill(xb, yb, c = 'w',zorder =3)
  ax[i, 0].plot(xb, yb, c = 'k', lw = 1, zorder = 5)
  ax[i, 1].plot(xb, yb, c = 'k', lw = 1, zorder = 5)
  ax[i,0].set_title(f"ROM No.{128 + no_eval}\n" + r"$E_k$" + f" = {np.round( E_k,decimals=2)}"+ r"$\%$")
  ax[i,1].set_title(f"DNS No.{128 + no_eval}")
  # ax[2].set_title("Absolute Error")
plt.subplots_adjust(wspace=0.05)
cax1 = fig.add_axes([ax[-1,0].get_position().x0,

                        ax[-1,0].get_position().y0-0.08,
                        ax[-1,0].get_position().width,
                        0.02])
cax2 = fig.add_axes([ax[-1,1].get_position().x0,

                        ax[-1,1].get_position().y0-0.08,
                        ax[-1,1].get_position().width,
                        0.02])
cbar1 = fig.colorbar(clb2, cax=cax1,orientation="horizontal")
cbar2 = fig.colorbar(clb2, cax=cax2,orientation="horizontal")
cbar1.set_ticks([-0.3,0,0.3])
cbar2.set_ticks([-0.3,0,0.3])
plt.savefig( save_snap_pred +  "Snapshot_" + fileID + ".jpg", bbox_inches="tight",dpi = 500)
print(f"The snap has been visualised")


EK_List   = []
for ind in range(cfg.in_dim,Physical_prediction.shape[0]):
  up      =   Physical_prediction[ind,0,:,:].T 
  ug      =   u_test[ind,0,:,:].T 
  E_k     =   1 - np.mean((up - ug )**2)/np.mean(ug **2 )
  E_k     *=  100 
  EK_List.append(E_k)

fig, axs  = plt.subplots(1,1,figsize=(8,3))
axs.plot(range(cfg.in_dim,Physical_prediction.shape[0]), EK_List)
axs.set_ylabel(r"$E_k$" + " (%)" , fontsize = 20)
axs.set_xlabel("Prediction Steps", fontsize = 20)
plt.savefig(save_fig_pred + "EK_" + fileID + ".jpg", bbox_inches= "tight")

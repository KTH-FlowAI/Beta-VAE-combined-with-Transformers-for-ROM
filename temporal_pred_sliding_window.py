"""
Train easy attention on the mu only 
"""

import  torch
from    utils.NNs.easyAttns import  easyTransformerEncoder
import  os
import  h5py
import  numpy               as      np
from    matplotlib          import  pyplot  as  plt
from    utils.configs       import  EasyAttn_config as cfg, VAE_custom, Name_Costum_VAE, Make_Transformer_Name
from    utils.pp            import Sliding_Window_Error
from    utils.plot          import colorplate as cc 
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


vae_name                =   Name_Costum_VAE(VAE_custom,nt= 26000)
base_dir                =   os.getcwd()
base_dir                +=  "/"
datafile                = base_dir +  '01_Data/u_1_to_26.hdf5'
checkpoint_save_path    = base_dir +  "06_ROM/OnlyPredictor/CheckPoints/"
CheckPoint_path         = base_dir +  "02_Checkpoints/"
modes_data_path         = base_dir +  "03_Mode/"
save_fig_pred           = base_dir +  f"04_Figs/vis_pred/dim{VAE_custom.latent_dim}/"
save_window_pred        = base_dir +  f"04_Figs/vis_pred/sliding_window/"
save_snap_pred          = base_dir +  f"04_Figs/vis_pred/snapshots/"
save_data_pred          = base_dir +  f"06_Preds/"


if args.model == "easy":
  from    utils.configs       import  EasyAttn_config as cfg, VAE_custom, Name_Costum_VAE, Make_Transformer_Name
  from    utils.NNs.easyAttns import  easyTransformerEncoder

  print("#"*30)
  fileID                  =   Make_Transformer_Name(cfg)
  vae_name                =   Name_Costum_VAE(VAE_custom,nt= 26000)
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
  vae_name                =   Name_Costum_VAE(VAE_custom,nt= 26000)
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

# Get the test data
Ntrain      =   int(cfg.train_split* nt)
u_test      =   u_keras[Ntrain: ,:,:]
print(f"Load the snapshots with shape = {u_test.shape}")
d           =   np.load(modes_data_path + vae_name + "modes.npz")
z_mean      =   d["z_mean"]
z_var       =   d["z_var"]
Nt          =   z_mean.shape[0]
Ntrain      =   int(cfg.train_split* Nt)
train_mean  =   z_mean[:Ntrain]
test_mean   =   z_mean[Ntrain:]
print(f"INFO: Load latent variable: Ntrain = {Ntrain}. train mean = {train_mean.shape}, test_mean = {test_mean.shape}")

train_var   =   z_var[:Ntrain,:]
test_var    =   z_var[Ntrain:,:]
test_data   =   test_mean

print("#"*30)
print("Examining Sliding window error")
window_size = 200
l2_error  = Sliding_Window_Error(test_data  = test_data, 
                                 model      = preditor, 
                                 device     = device,
                                 in_dim     = cfg.in_dim,
                                 window     = window_size)


fig, axs  =  plt.subplots(1,1,figsize = (8, 4)) 
axs.plot(l2_error,lw = 2 , c =cc.blue)
axs.set_ylabel(r"${\epsilon}$", fontsize = 20)
axs.set_xlabel("Prediction Steps")
plt.savefig(save_window_pred + "Window_" + fileID + ".jpg", bbox_inches= "tight")



np.savez_compressed(save_data_pred +"Pred_Window_" + fileID + ".npz", 
                    
                    window_err  =  l2_error, 
                    window_size =  window_size,
                  
                    )


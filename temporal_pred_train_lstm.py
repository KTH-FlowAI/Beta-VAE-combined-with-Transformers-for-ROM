"""
Train easy attention on the mu only 
@author yuningw
Sep 25
"""

import  torch
from    utils.NNs.RNNs     import LSTMs
import  os
import  h5py
import  time
import  numpy               as      np
from    matplotlib          import  pyplot  as  plt
from    utils.configs       import  LSTM_config as cfg, VAE_custom, Name_Costum_VAE, Make_LSTM_Name
from    utils.datas         import make_DataLoader, make_Sequence 
from    utils.pp            import make_Prediction
from    utils.plot          import plot_signal
from    utils.train         import fit

torch.manual_seed(1024)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
#confirm device
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

fileID                  =   Make_LSTM_Name(cfg)
vae_name                =   Name_Costum_VAE(VAE_custom,nt= 25999)
base_dir                =  "/mimer/NOBACKUP/groups/deepmechalvis/yuningw/Cylinder_ROM/"
checkpoint_save_path    = base_dir +  "06_ROM/OnlyPredictor/CheckPoints/"
modes_data_path         = base_dir +  "03_Mode/"
save_fig_pred           = base_dir +  f"04_Figs/vis_pred/dim{VAE_custom.latent_dim}/"
save_data_pred          = base_dir +  f"06_Preds/dim{VAE_custom.latent_dim}/"
fileID                  = "Mean_" +  fileID + "_" + vae_name 
print(f"INFO: the fileID is\n{fileID}")

d           =   np.load(modes_data_path + vae_name + "modes.npz")
z_mean      =   d["z_mean"]
z_var       =   d["z_var"]

Nt          =   z_mean.shape[0]
Ntrain      =   int(cfg.train_split* Nt)
print(z_mean.shape)
# quit()
train_mean  =   z_mean[:Ntrain]
test_mean   =   z_mean[Ntrain:]


print(f"INFO: Ntrain = {Ntrain}. train mean = {train_mean.shape}, test_mean = {test_mean.shape}")
train_var   =   z_var[:Ntrain,:]
test_var    =   z_var[Ntrain:,:]


test_data   =    test_mean + np.exp(0.5 * test_var) * np.random.random(size = test_var.shape)
train_data  =    z_mean[:Ntrain,:]
test_data   =    z_mean[Ntrain:,:]
# train_data       =    train_mean + np.exp(0.5 * train_var) * np.random.random(size = train_mean.shape)
print(train_data.shape)

# quit()
TInterval   =    int(1)
train_data  = train_data[::TInterval]
test_data   = test_data[::TInterval]

print(f"Training data = {train_data.shape}, test data = {test_data.shape}")
# Generate the Training Data and DataLoader
X, Y = make_Sequence(cfg=cfg, data=train_data)

train_dl, val_dl = make_DataLoader(torch.from_numpy(X),torch.from_numpy(Y),
                                   batch_size=cfg.Batch_size,
                                   drop_last=False, train_split=cfg.train_split)

print(f"INFO: The DataLoaders made, num of batch in train={len(train_dl)}, validation={len(val_dl)}")
## Examine the input shape 
x,y = next(iter(train_dl))
print(f"Examine the input and output shape = {x.shape, y.shape}")

model = LSTMs(
                d_input= cfg.in_dim, d_model= cfg.d_model, nmode= cfg.nmode,
                embed= cfg.embed, hidden_size= cfg.hidden_size, num_layer= cfg.num_layer,
                is_output= cfg.is_output, out_dim= cfg.next_step, out_act= cfg.out_act
                )

NumPara = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"INFO: The model has been generated, num of parameter is {NumPara}")

## Compile 
loss_fn = torch.nn.MSELoss()
opt     = torch.optim.Adam(model.parameters(),lr = cfg.lr, eps=1e-7)
opt_sch = [  
            torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma= (1 - 1/cfg.Epoch)) 
            ]
# opt_sch = None
# Training 
s_t     = time.time()
history = fit(device, model, train_dl, 
           loss_fn,cfg.Epoch,opt,val_dl, 
           scheduler=opt_sch,if_early_stop=cfg.early_stop,patience=cfg.patience)
e_t     = time.time()
cost_time = e_t - s_t
print(f"INFO: Training ended, spend time = {np.round(cost_time)}s")
# Save Checkpoint
check_point = {"model":model.state_dict(),
               "history":history,
               "time":cost_time}
torch.save(check_point,checkpoint_save_path+ fileID + ".pt")
print(f"INFO: The checkpoints has been saved!")

# Make prediction on test data
Preds = make_Prediction(test_data, model,device,
                        in_dim= cfg.in_dim,
                        next_step= cfg.next_step)
print(f"INFO: Prediction on TEST has been generated!")


plot_signal(test_data= test_data, Preds= Preds, 
            save_file= save_fig_pred +\
                    "Pred_" + fileID + ".jpg",)



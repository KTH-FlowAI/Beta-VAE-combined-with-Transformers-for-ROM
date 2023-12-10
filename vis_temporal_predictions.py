"""
A script to reproduce the time-series prediction results
@yuningw
"""

"""
Visualisation of the post processing of temporal-prediction
"""

import  os
import  h5py
import  numpy               as      np
from    matplotlib          import  pyplot  as  plt
from    utils.plot          import colorplate as cc 
from    utils               import plt_rc_setup
from    utils.configs       import VAE_custom 

base_dir                =   os.getcwd()
base_dir                +=  "/"
load_data_pred          = base_dir +  f"05_Pred/"
Zdim                    = VAE_custom.latent_dim


easy_ID                 =  "easyattn"
self_ID                 =  "selfattn"
lstm_ID                 =  "LSTM"

fileIDs                 =  [
                          
                            easy_ID, 
                            self_ID, 
                            lstm_ID,
                            ]

####################################################

Colors                  =   [
                             cc.black, 
                             cc.red, 
                             cc.blue, 
                             cc.cyan,
                             ]
###################################################

print("#"*30)



## Load the prediction
window_Errors   =   []
Snapshots       =   []
Snapshots_g     =   []
Signals         =   []

SnapNos         =    np.array([1+128, 26 + 128, 51+ 128])

d               =   np.load(load_data_pred + "Timeseries_true.npz")
test_signal     =   d['timeseries']
d               =   np.load(load_data_pred + "Snapshots_true.npz")
test_snapshot   =   d['snapshots']

         

Signals.append(test_signal)
Snapshots.append(test_snapshot)


for  i, fileID in enumerate(fileIDs):

    d           =   np.load(load_data_pred +"Snapshots_" + fileID + ".npz")
    snap_p      =   d['snapshots']

    d           =   np.load(load_data_pred +"Timeseries_" + fileID + ".npz")
    signal_p    =   d['timeseries']

    del d 
    d           =   np.load(load_data_pred+ "Sliding_window_"+ fileID + ".npz")

    window_Errors.append(d["window_err"])
    Signals.append(signal_p)
    Snapshots.append(snap_p)


Labels          = [ 
                    "True",
                    "Easy-Attn",
                    "Self-Attn",
                    "LSTM",
                    ]


fig, axs = plt.subplots(Zdim,1, figsize= (8, 8), sharex= True)
for i,  signal in enumerate(Signals): 
   
   for j in range(Zdim):
        if j == -1:

            axs[j].plot(signal[:,j], lw = 2, 
                    c = Colors[i], label = Labels[i],alpha = 1)
        else:
            axs[j].plot(signal[:,j], lw = 2, 
                    c = Colors[i], label = Labels[i],alpha = 1)
        axs[j].set_ylim(-2.8, 2.8)
        axs[j].spines['top'].set_visible(False)
        axs[j].spines['right'].set_visible(False)
        
        if j == Zdim-1:
            axs[j].text(0.88, 0.12, 'Mode '+str(j+1), fontsize=15, transform=axs[j].transAxes, bbox=dict(facecolor='white', alpha=0.2))
        else:
            axs[j].text(0.9, 0.12, 'Mode '+str(j+1), fontsize=15, transform=axs[j].transAxes, bbox=dict(facecolor='white', alpha=0.2))
         
        printtrue = False

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.92))
axs[-1].set_xlabel("Prediction steps",fontsize = 20)
plt.subplots_adjust(hspace= 0.25)

plt.savefig(f"04_Figs/Temporal/Singal_compare.pdf", bbox_inches="tight", dpi =1000)

##########
print("#"*30)
print("INFO: Visualisation of Sliding window")

fig, axs = plt.subplots(1,1, figsize = (8,4))

for i in range(len(fileIDs)):
    window_err = window_Errors[i]
    axs.plot(window_err, lw = 3, c = Colors[i+1])
axs.set_xlabel("Prediction steps",fontsize = 20)
axs.set_ylabel(r"$\epsilon$",fontsize = 20)    
plt.legend(Labels[1:], ncol = 3,loc = "upper center", bbox_to_anchor = (0.5,1.15))
plt.savefig(f"04_Figs/Temporal/Window_compare.pdf", bbox_inches="tight", dpi =1000)

# quit()
Labels          = [ 
                    "DNS",
                    "Easy-Attn",
                    "Self-Attn",
                    "LSTM",
                    ]

print("#"*30)
print("Visualsize Reconstruction")
Nx, Ny  =   192, 96 
x       =   np.linspace(-1, 5, Nx)
y       =   np.linspace(-1.5, 1.5, Ny)
y, x    =   np.meshgrid(y, x)
x       =   x[:192, :96]
y       =   y[:192, :96]
xb      =   np.array([-0.15, -0.15, 0.15,  0.15, -0.15])
yb      =   np.array([-0.15,  0.15, 0.15, -0.15, -0.15])
vmin    =   np.concatenate(Snapshots).min()
vmax    =   np.concatenate(Snapshots).max()

print(len(Snapshots),len(SnapNos))

fig, axs = plt.subplots(    len(Snapshots), 
                            SnapNos.shape[0],
                            sharex = True, 
                            sharey = True,
                            figsize = (12, 12)
                        )

contourf_plots = []
for i in range(len(Snapshots)):

    for j in range(len(SnapNos)):
        
        clb=axs[i,j].contourf(  
                            x,y, 
                            Snapshots[i][j,0,:,:].T,
                            levels = 200, 
                            vmin = vmin, vmax = vmax,
                            cmap= "YlGnBu_r"
                          )
        
        axs[-1, j].set_xlabel("x/h")
        axs[i, 0].set_ylabel("z/h")
        axs[i, j].fill(xb, yb, c = 'w',zorder =3)
        axs[i, j].fill(xb, yb, c = 'w',zorder =3)
        axs[i, j].plot(xb, yb, c = 'k', lw = 1, zorder = 5)
        axs[i, j].plot(xb, yb, c = 'k', lw = 1, zorder = 5)

        axs[i,j].set_aspect("equal")   

        if  i!= 0:
            up = Snapshots[i][j,0,:,:].T
            ug = Snapshots[0][j,0,:,:].T

            Ek = (1 -  (np.mean((up-ug)**2)/np.mean(ug**2)) )*100

            axs[i,j].set_title(Labels[i]+"\n" + r"$E_k$" + " = " + f"{np.round(Ek,3)}%") 
        
        else:
            if j == 0:
                No_snap = SnapNos[j]-128
                axs[i,j].set_title("t"+ " + " + r"$\Delta$" + "t" + "\n" + Labels[i]) 

            else:
                No_snap = SnapNos[j]-128-1
                axs[i,j].set_title("t"+ " + " +  f"{No_snap}" + r"$\Delta$" + "t" + "\n" + Labels[i]) 
        contourf_plots.append(clb)


colorbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.68])  # [left, bottom, width, height]

# Create the colorbar using the contourf objects
cbar = plt.colorbar(contourf_plots[0], cax=colorbar_ax)
plt.savefig(f"04_Figs/Temporal/Snapshots.jpg", bbox_inches='tight', dpi = 500)


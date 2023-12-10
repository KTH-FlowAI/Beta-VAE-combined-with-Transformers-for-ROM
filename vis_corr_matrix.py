"""
The script for reproducing the correlation matrix 
@yuningw
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors

################
#Color Composition to play with
################

colorlist= ["#A64036","#F0C2A2","#4182A4","#354E6B"]
colorlist.reverse()
china_color = colors.LinearSegmentedColormap.from_list('china_color',colorlist,N=100)
plt.register_cmap(cmap=china_color)

plt.set_cmap(china_color)

font_dict = {"weight":"bold","size":22}
plt.rc("font",family = "serif")
plt.rc("font",size = 14)
plt.rc("axes",labelsize = 16, linewidth = 2)
plt.rc("legend",fontsize= 12, handletextpad = 0.3)
plt.rc("xtick",labelsize = 14)
plt.rc("ytick",labelsize = 14)


vae_types = ['v2','v3']
num_fields = 10000
latent_dim = 10
betas = [0.001,0.0025,0.005,0.01]
Epoch = 300

for vae_type in vae_types:

    corrs = []
    Dets = []
    for beta in betas:
        filesID = vae_type+"_n"+str(num_fields)+'_m'+str(latent_dim)+'_b'+str(int(beta*10000))+"e-4" +"_epoch" + str(Epoch)
        modes_filepath ="03_Mode/modes/"+filesID +"_modes"+ ".hdf5"

        print(filesID)

        d =  h5py.File(modes_filepath,"r")
        print(d.keys())

        z_mean = np.array(d["z_mean"])

        corr_matrix_latent = abs(np.corrcoef(z_mean.T))
        detR = np.linalg.det(corr_matrix_latent)
        print(f"In order to confirm the case ,the detR is {np.round(detR,4)}")
        corrs.append(corr_matrix_latent)
        Dets.append(detR)

    betas_label = [r"$1\times10^{-3}$", r"$2.5\times10^{-3}$",r"$5\times10^{-3}$", r"$1\times10^{-2}$"  ]

    fig, axs = plt.subplots(1,4,figsize=(12, 8),sharey=True)
    axs = axs.flatten()
    for ind, ax in enumerate(axs):
        corr_matrix_latent = corrs[ind]
        cb = ax.imshow(corr_matrix_latent)
        ax.set_title(r"${\beta}$"+ " = " + betas_label[ind]+"\n" + r"${{\rm det}_{\mathbf{R}}}$" +f" = {np.round(100*Dets[ind],2)}")
        ax.set_xticks(range(latent_dim))
        ax.set_yticks(range(latent_dim))
        ax.set_xticklabels(range(1,latent_dim+1))
        ax.set_yticklabels(range(1,latent_dim+1))
        ax.set_xlabel(r"$z_i$",fontdict = font_dict )
    axs[0].set_ylabel(r"$z_i$",fontdict = font_dict )
    cax = fig.add_axes([axs[3].get_position().x1+0.03,axs[3].get_position().y0,0.02,0.25])
    cbar = fig.colorbar(cb, cax=cax)
    cbar.ax.locator_params(nbins = 5,tight=True)
    plt.savefig("04_Figs/CorrMatrix/Corr_"+vae_type+"_b_all"+f"_{latent_dim}m"+f"_{num_fields}n_8"+".pdf",bbox_inches="tight",dpi=300)
"""
The script for generating ranked spatial modes 
@yuningw
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

cmp = "RdBu"
plt.set_cmap(cmp)
plt.rc('font', family = 'serif')
plt.rc('axes', labelsize = 22, linewidth = 1.5)
plt.rc('font', size = 20)
plt.rc('legend', fontsize = 12, handletextpad=0.3)
plt.rc('xtick', labelsize = 25)
plt.rc('ytick', labelsize = 25)

num_fields = 10000
latent_dim = 10
Epoch      = 300
vae_type   = "v3"
vae_types  = ['v2','v3']
betas      = [0.001,0.0025,0.005,0.01]

print("#"*30)
print("INFO: Start Plotting the spatial Modes of POD and VAE")

for vae_type in vae_types:
    modes = []
    temporals = []
    for beta in betas:
        filesID = vae_type+"_n"+str(num_fields)+'_m'+str(latent_dim)+'_b'+str(int(beta*10000))+"e-4" +"_epoch" + str(Epoch)
        modes_filepath = "03_Mode/ranked/"+filesID +"_rank"+ ".npz"

        print(filesID)
        d =  np.load(modes_filepath)
        z_mean = d["z_mean"]

        corr_matrix_latent = abs(np.corrcoef(z_mean.T))
        detR = np.linalg.det(corr_matrix_latent)
        print(f"In order to confirm the case ,the detR is {np.round(detR,4)}")

        m = np.array(d["modes"])
        # m *= u_std
        print(m.shape)
        ran = d["m"]
        print(f"The order is {ran}\n")
        modes.append(m)
        z_mean_rank = z_mean[:,ran]
        temporals.append(z_mean_rank)

    # %%
    pod_file = f"08_POD/POD-m{latent_dim}-n{num_fields}.npz"
    pod = np.load(pod_file)
    U = pod["modes"]
    vh = pod["vh"]
    print(vh.shape)

    print(U.min(),U.max())
    cnv = np.sqrt(num_fields-1)
    U *= cnv
    U  = np.expand_dims(U,-1)
    print(U.shape)
    print(U.min(),U.max())
    modes.append(U)
    vh *= cnv
    temporals.append(vh.T)
    v_max = np.round(np.array(modes).max(),2)
    v_min = np.round(-np.array(modes).max(),2)
    print(np.array(modes).shape)



    betas_label = [r"$1\times10^{-3}$", r"$2.5\times10^{-3}$",r"$5\times10^{-3}$", r"$1\times10^{-2}$"  ]
    print(len(modes), m.shape[0])
    x = np.linspace(-1, 5, 192)
    y = np.linspace(-1.5, 1.5, 96)
    y, x = np.meshgrid(y, x)
    x = x[:192, :96]
    y = y[:192, :96]

    xb = np.array([-0.15, -0.15, 0.15, 0.15, -0.15])
    yb = np.array([-0.15, 0.15, 0.15, -0.15, -0.15])

    fig,ax = plt.subplots(latent_dim,len(modes),
                            sharex=True,sharey=True,
                            figsize=(3.5*len(modes),4*len(modes)))
    annhight = 1.2

    for j in tqdm(range(len(modes))):

        ax[-1,j].set_xlabel(r"${x/h}$")
        v_max = np.round(np.array(modes)[j,:,:,:,:].max(),4)
        v_min = np.round(-np.array(modes)[j,:,:,:,:].max(),4)
        for i in range(m.shape[0]):
            cb=ax[i,j].contourf(x,y,modes[j][i,:,:,0].T,levels=200, vmin= v_min,vmax = v_max)
            ax[i,j].fill(xb, yb, c = 'w',zorder =3)
            ax[i,j].plot(xb, yb, c = 'k', lw = 1, zorder = 5)
            ax[i,j].set_aspect('equal')
            ax[i,j].set_title(f"Mode {i+1}")
            ax[i,0].set_ylabel(r"${z/h}$")

            if j != len(modes)-1:
                ax[0,j].annotate( r"${\beta}$" + " = " + betas_label[j], xy=(0.5, annhight), xytext=(0, 5),
                                        xycoords='axes fraction', textcoords='offset points',
                                        ha='center', va='baseline', fontsize = 25)
            else:
                ax[0,j].annotate(f"POD", xy=(0.5, annhight), xytext=(0, 5),
                                        xycoords='axes fraction', textcoords='offset points',
                                        ha='center', va='baseline', fontsize = 25)

    plt.tight_layout()

    plt.savefig(f"04_Figs/Modes/Modes_Ranked_{vae_type}_{num_fields}n_{latent_dim}m_5" + ".jpg",bbox_inches="tight",dpi=300)
    print(f"INFO: Spatial modes of Arch {vae_type} finished!")
print("#"*30)
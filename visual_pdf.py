"""
A script for reproducing the PDFs 
@yuningw
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
red = "#D12920" #yinzhu
blue = "#2E59A7" # qunqing
gray = "#DFE0D9" # ermuyu
plt.rc('font', family = 'serif')
plt.rc('axes', labelsize = 20, linewidth = 1.5)
plt.rc('font', size = 18)
plt.rc('legend', fontsize = 12, handletextpad=0.3)
plt.rc('xtick', labelsize = 21)
plt.rc('ytick', labelsize = 21)
font_dict = {"weight":"bold","size":20}

num_fields = 10000
latent_dim = 10
betas = [0.001,0.0025,0.005,0.01]
Epoch = 300
vae_types = ["v2","v3"]

FigDir = "04_Figs/pdfPlot/"

print("#"*30)
print("INFO: Start generating PDFs")
print("The effect of beta:")

for vae_type in vae_types:
    modes = []
    for beta in betas:
        filesID = vae_type+"_n"+str(num_fields)+'_m'+str(latent_dim)+'_b'+str(int(beta*10000))+"e-4" +"_epoch" + str(Epoch)
        modes_filepath = "03_Mode/modes/"+filesID + "_modes" + ".hdf5"
        d =  h5py.File(modes_filepath,"r")
        z_mean = np.array(d["z_mean"])
        corr_matrix_latent = abs(np.corrcoef(z_mean.T))
        detR = np.linalg.det(corr_matrix_latent)
        print(f"In order to confirm the case ,the detR is {np.round(detR,4)}")
        modes.append(z_mean)

    Inds = []
    for m in modes:
        std = np.std(m,axis=0)
        ind = std > 0.5
        Inds.append(ind)

    fig, axs = plt.subplots(1, len(modes),
                            sharex=True, sharey= True,
                            figsize =(4*len(modes),3)
                            )

    betas_label = [r"$1\times10^{-3}$", r"$2.5\times10^{-3}$",r"$5\times10^{-3}$", r"$1\times10^{-2}$"  ]
    axs = axs.flatten()
    for j, ax in enumerate(axs):
        z_mean = modes[j]
        kdes = []
        for i in range(z_mean.shape[1]):
            ri = z_mean[:,i]
            xx = np.linspace(-2.5,2.5,z_mean.shape[0])
            pdf    = gaussian_kde(ri)
            kdes.append(pdf(xx))
        kdes = np.array(kdes)
        inds = Inds[j]
        for i in range(z_mean.shape[1]):
            if inds[i]:
                ax.plot(xx, kdes[i]/kdes[i].max(),zorder= 3,c = blue)
                ax.fill_between(xx, kdes[i]/kdes[i].max(),zorder= 3,color = blue,alpha =0.15)
            else:
                ax.plot(xx, kdes[i]/kdes[i].max(),zorder= 5,c = red)
        n_noise = inds.shape[0] -  np.count_nonzero(inds)
        ax.set_xlim(-2.5,2.5)
        ax.set_ylim(0,1.2)
        ax.set_title(r"${\beta}$"+" = "+ betas_label[j] , font_dict)
        print(f"For beta =  {betas[j]}, there are {n_noise} noises")
        ax.set_xlabel(r"${z_i}$",fontdict = font_dict)
    axs[0].set_ylabel("PDF",fontdict= font_dict)

    plt.savefig(FigDir +"PDF_4_"+vae_type+"_n"+str(num_fields)+'_m'+str(latent_dim)+'_b_all'+"_epoch" + str(Epoch)+".jpg",
                bbox_inches = "tight",dpi= 500)
    
    print(f"INFO: {vae_type} PDF for Beta effect FINISHED!")


print("#"*30)
print("The effect of Latent Dim")

num_fields = 10000
latent_dims = [5,10,20]
beta = 0.005
Epoch = 300


for vae_type in vae_types:
    modes = []
    for latent_dim in latent_dims:
        filesID = vae_type+"_n"+str(num_fields)+'_m'+str(latent_dim)+'_b'+str(int(beta*10000))+"e-4" +"_epoch" + str(Epoch)
        modes_filepath = "03_Mode/modes/"+filesID + "_modes" + ".hdf5"
        d =  h5py.File(modes_filepath,"r")
        z_mean = np.array(d["z_mean"])
        corr_matrix_latent = abs(np.corrcoef(z_mean.T))
        detR = np.linalg.det(corr_matrix_latent)
        print(f"In order to confirm the case ,the detR is {np.round(detR,4)}")
        modes.append(z_mean)
    Inds = []
    for m in modes:
        std = np.std(m,axis=0)
        ind = std > 0.5
        Inds.append(ind)
    fig, axs = plt.subplots(1, 3,
                            sharex=True, sharey= True,
                            figsize = (4*len(modes), 3))
    font_dict = {"weight":"bold","size":22}
    axs = axs.flatten()
    for j, ax in enumerate(axs):
        z_mean = modes[j]
        kdes = []
        for i in range(z_mean.shape[1]):
            ri = z_mean[:,i]
            xx = np.linspace(-2.5,2.5,z_mean.shape[0])
            pdf    = gaussian_kde(ri)
            kdes.append(pdf(xx))
        kdes = np.array(kdes)
        inds = Inds[j]
        for i in range(z_mean.shape[1]):
            if inds[i]:
                ax.plot(xx, kdes[i]/kdes[i].max(),zorder= 3,c = blue)
                ax.fill_between(xx, kdes[i]/kdes[i].max(),zorder= 3,color = blue,alpha =0.2)
            else:
                ax.plot(xx, kdes[i]/kdes[i].max(),zorder= 5,c = red)
        n_noise = inds.shape[0] -  np.count_nonzero(inds)
        print(f"For Latent dim =  {latent_dims[j]}, there are {n_noise} noises")
    
        ax.set_xlim(-2.5,2.5)
        ax.set_ylim(0,1.2)
        axs[0].set_ylabel("PDF",font_dict)
        ax.set_xlabel("$z_i$",font_dict)
        ax.set_xticks([-2,0,2])
        ax.set_title(r"$d$" +" = " + f"{latent_dims[j]}",font_dict)
    plt.savefig(FigDir+"PDF_"+vae_type+f"_n_{num_fields}"+'_m_all'+'_b'+str(int(beta*10000))+"e-4"+"_epoch" + str(Epoch)+".jpg",
                bbox_inches = "tight",dpi=500)
    
    print(f"INFO: {vae_type} PDF for Latent Dim effect FINISHED!")

print("END")
print("#"*30)
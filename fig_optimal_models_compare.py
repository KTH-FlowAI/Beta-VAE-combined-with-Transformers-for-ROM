"""
Reproduce the figure of optimal model results comparsion
@yuningw 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import json
red   = "#D12920" #yinzhu
blue  = "#2E59A7" # qunqing
black = "#151D29" # lanjian
plt.rc('font', family = 'serif')
plt.rc('axes', labelsize = 20, linewidth = 1.5)
plt.rc('font', size = 18)
plt.rc('legend', fontsize = 12, handletextpad=0.3)
plt.rc('xtick', labelsize = 21)
plt.rc('ytick', labelsize = 20)

font_dict = {"weight":"bold","size":22}

num_fields  = 10000
latent_dim  = 10
Epoch       = 1000
vae_types   = ["v2","v3"]
beta        = 0.005
temporals   = {}
spatials    = {}
for vae in vae_types:
  temporals[vae] = []
  spatials[vae] = []

for vae_type in vae_types:

      filesID        = vae_type+"_n"+str(num_fields)+'_m'+str(latent_dim)+'_b'+str(int(beta*10000))+"e-4" +"_epoch" + str(Epoch)
      modes_filepath ="03_Mode/ranked/"+filesID +"_rank"+ ".npz"
      print(filesID)
      d =  np.load(modes_filepath)
      z_mean = d["z_mean"]
      print(z_mean.shape)
      corr_matrix_latent = abs(np.corrcoef(z_mean.T))
      detR = np.linalg.det(corr_matrix_latent)
      print(f"In order to confirm the case ,the detR is {np.round(detR,4)}")

      m = np.array(d["modes"])
      spatials[vae_type].append(m)
      ran = d["m"]
      print(f"The order is {ran}\n")
      z_mean_rank = z_mean[:,ran]
      temporals[vae_type].append(z_mean_rank)

# %%
pod_file = f"08_POD/POD-m{latent_dim}-n{num_fields}.npz"
pod = np.load(pod_file)
U = pod["modes"]
vh = pod["vh"]
cnv = np.sqrt(num_fields-1)
U  = np.expand_dims(U,-1)
vh *= cnv

temporals["pod"] = []
temporals["pod"].append(vh.T)

spatials["pod"] = []
spatials["pod"].append(U)

corr_matrix_latent = abs(np.corrcoef(vh))
detR = np.linalg.det(corr_matrix_latent)
print(f"In order to confirm the case ,the detR is {np.round(detR,4)}")


temporal_spectrum = {}
names = ["v2","v3","pod"]
for name in names:
  temporal_spectrum[name] = np.array(temporals[name])
  ys = np.zeros(shape = (temporal_spectrum[name].shape[0],latent_dim))
  xs = np.zeros(shape= (temporal_spectrum[name].shape[0],latent_dim))

  fs = 1
  for i in range(temporal_spectrum[name].shape[0]):
    for j in range(latent_dim):
        f, Pxx_den = signal.welch(temporal_spectrum[name][i,:,j], fs, nperseg=4096)
        f /= 2*np.pi
        xs[i,j] = f[np.argmax(Pxx_den)]

        ys[i,j] = Pxx_den.max()
  temporal_spectrum[name] = xs



def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text = r"$f/{f_s}$ = "+"{:5f}".format(xmax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops,
              bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)


fig, axs = plt.subplots(2,3,
                        figsize = (16,6))
annhight = 1.2

for i in range(3):
  signals = np.array(temporals[names[i]]).squeeze()
  f, Pxx_den = signal.welch(signals[:,0], fs, nperseg=4096)

  f /= 0.005 # sampling frequency
  axs[0,i].plot(f, Pxx_den,lw =2.5, c = blue)
  axs[0,i].set_xlim([0, 1])
  axs[0,i].set_yticks([500, 1000])
  axs[0,i].set_xlabel(r"${St}$")

  axs[0,i].set_title(f"Mode 1")
  axs[0,i].set_ylim([0,1200])
  if i > 0:
    axs[0,i].yaxis.set_visible(False)
  annot_max(f,Pxx_den,ax = axs[0,i])

  if i < (len(temporals) -1):
              axs[0,i].annotate(f"{vae_types[i]}", xy=(0.5, annhight), xytext=(0, 6),
                                      xycoords='axes fraction', textcoords='offset points',
                                      ha='center', va='baseline', fontsize = 25)
  else:
              axs[0,i].annotate(f"POD", xy=(0.5, annhight), xytext=(0, 6),
                                      xycoords='axes fraction', textcoords='offset points',
                                      ha='center', va='baseline', fontsize = 25)



x = np.linspace(-1, 5, 192)
y = np.linspace(-1.5, 1.5, 96)
y, x = np.meshgrid(y, x)
x = x[:192, :96]
y = y[:192, :96]


xb = np.array([-0.15, -0.15, 0.15, 0.15, -0.15])
yb = np.array([-0.15, 0.15, 0.15, -0.15, -0.15])


for i in range(3):
  mode  = np.array(spatials[names[i]]).squeeze()
  v_min = mode.min()
  v_max = mode.max()
  cb=axs[1,i].contourf(
                            x,y,mode[0,:,:].T,levels=100,
                            vmin= v_min,vmax = v_max,
                              cmap= "RdBu"
                            )
  axs[1,i].fill(xb, yb, c = 'w',zorder =3)
  axs[1,i].plot(xb, yb, c = 'k', lw = 1, zorder = 5)
  axs[1,i].set_aspect('equal')
  axs[1,0].set_ylabel(r"${z/h}$")
  axs[1,i].set_xlabel(r"${x/h}$")
  if i > 0:
    axs[1,i].yaxis.set_visible(False)

plt.subplots_adjust(hspace = 0.6)
plt.savefig(f"04_Figs/Modes/Temp_Spat_Ranked_v2v3_{num_fields}n_{latent_dim}m_shedding" + ".jpg",
            bbox_inches="tight",dpi=800)


plt.rc("font",family = "serif")
plt.rc("font",size = 16)
plt.rc("axes",labelsize = 18, linewidth = 2)
plt.rc("legend",fontsize= 12, handletextpad = 0.3)
plt.rc("xtick",labelsize = 18)
plt.rc("ytick",labelsize = 18)
font_dict = {"weight":"bold","size":22}

num_fields = 10000
latent_dim = 10
Epoch = 1000
vae_types = ["v2","v3"]
beta = 0.005
hists = []
for vae_type in vae_types:
    filesID =  vae_type+"_n"+str(num_fields)+'_m'+str(latent_dim)+'_b'+str(int(beta*10000))+"e-4" +"_epoch" + str(Epoch)
    print(filesID)
    loss_file = "07_Loss/"+"loss_" + filesID + ".json"
    print(loss_file)

    with open(loss_file,"r") as f:

      history = json.load(f)

      hists.append(history)
    f.close()

fig, ax = plt.subplots(1,1,sharex=True,figsize = (12,4))
ax2 = ax.twinx()

ax.tick_params("y",colors = blue)
ax.yaxis.label.set_color(blue)
ax2.tick_params("y",colors = red)
ax2.yaxis.label.set_color(red)
ax2.spines["left"].set_color(blue)
ax2.spines["right"].set_color(red)

ax.plot(hists[0]["reconstruction_loss"],"-",c= blue,lw=2.5,label= f"{vae_types[0]} Total")
ax2.plot(hists[0]["kl_loss"],"-",c = red,lw=2.5,label= f"{vae_types[0]} Kl loss")
ax.plot(hists[1]["reconstruction_loss"],"--",c= blue,lw=2.5,label= f"{vae_types[1]} Total")
ax2.plot(hists[1]["kl_loss"],"--",c = red,lw=2.5,label= f"{vae_types[1]} Kl loss")
ax.set_yscale("log")
ax.set_ylim([0,0.1])
ax.set_ylabel( r"${\mathcal{L}_{\rm rec}}$", font_dict)
ax2.set_ylabel(r"$D_{KL}$", )
ax.set_xlabel("Train Step", font_dict)
plt.savefig(f"04_Figs/Loss/Loss_v2v3__{num_fields}n_{latent_dim}m_{Epoch}Epoch.jpg",
            bbox_inches="tight",dpi=1000)
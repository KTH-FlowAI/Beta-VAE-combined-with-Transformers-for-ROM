"""
Script for POD using snapshot-POD method 
"""
import numpy as np 
import h5py
import argparse
import time 
import os 
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

parser      =   argparse.ArgumentParser(description='POD')
parser.add_argument("--full","-f", action="store_true",help="compute the full rank of POD")
parser.add_argument("--rank","-r", default=10, type=int, help='number of modes')
args        =   parser.parse_args()

datafile    =   '01_Data/u_1_to_13.hdf5'
podfile     =   "08_POD/"
with h5py.File( datafile, 'r') as f:
  u_keras   =   f['u'][:]
  nt        =   f['nt'][()]
  nx        =   f['nx'][()]
  nz        =   f['nz'][()]
  u_mean    =   f['mean'][:]
  u_std     =   f['std'][:]

u_keras     =   np.nan_to_num(u_keras)
u_keras     =   np.transpose( u_keras, (0,2,1))
u_keras     =   u_keras[:,:,:,np.newaxis]

u           =   u_keras * u_std
print(f"INFO:The mean velocity has been decomposed")
nt, nx, nz, nc = u.shape

# POD using SVD
s_t         =   time.time() 
u           =   u.reshape((nt, -1)).T
cov         =   np.sqrt(nt-1)
u           =   u/cov
U, s, vh    =   np.linalg.svd(u,full_matrices=False)
e_t         =   time.time()
c_t         =   e_t - s_t
print(f"INFO: SVD finished, spend time {c_t}s")

if args.full: 
    r         =   -1
    print(f"INFO: Going to use FULL modes for reconstruction")

else: 
    r         =   args.rank
    print(f"INFO: Going to use {r} modes for reconstruction")


U           =   U[:, :r]
s           =   s[:r]
vh          =   vh[:r]

u_p         =   U @ np.diag(s) @ vh
u_p        *=   cov
u          *=   cov
u           =   u.T.reshape((-1, nx, nz,nc))
u_p         =   u_p.T.reshape((-1, nx, nz,nc))
U           =   U.T.reshape((-1, nx, nz))

err = np.mean((u - u_p)**2, axis = (0 ,1, 2))/np.mean(u**2, axis = (0 ,1, 2))
energy = (1 - err)*100

print(f"Use mode = {r}, the energy level is {np.round(energy,3)}%")

np.savez_compressed( podfile +f"POD-m{r}-n{nt}"+".npz",
                    modes = U,
                    vh    = vh,
                    s     = s,
                    t     = c_t,
                    )
print("INFO: Data has been saved!")



import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.ticker import LogLocator, NullFormatter

red     = "#D12920" #yinzhu
blue    = "#2E59A7" # qunqing
gray    = "#DFE0D9" # ermuyu 


plt.rc("font",family = "serif")
plt.rc("font",size = 16)
plt.rc("axes",labelsize = 18, linewidth = 2)
plt.rc("legend",fontsize= 12, handletextpad = 0.3)
plt.rc("xtick",labelsize = 16)
plt.rc("ytick",labelsize = 18)


print(s.shape)
nr  =   s.shape[0]
r   =   np.arange(nr)
s   =   s**2
s1  =   (s/np.sum(s))[:nr]
s2  =   (np.cumsum(s)/np.sum(s))[:nr]
r99 =   np.min(np.where(s2 > 0.99)[0])

print(f"{r99} of eigenvalue to reconstruct 99 energy")

fig, ax = plt.subplots(1,2,sharex=True,figsize = (12,4))
locmin = LogLocator(base=1.0,subs=np.arange(1,10)*0.1, numticks=100)

ax[0].xaxis.set_minor_locator(locmin)
ax[0].xaxis.set_minor_formatter(NullFormatter())
ax[0].plot(r,s1,c =blue ,lw = 2.5)
ax[0].set_xscale("log")
ax[0].set_ylabel(r"$\lambda_i / \sum_{ j = 1 }^{ j = N_t } \lambda_j$")
ax[0].set_ylim(1,nr)
ax[0].set_ylim(0,0.08)

ax[1].plot(r,s2,c = blue,lw = 2.5)
ax[1].set_ylabel(r"$\sum_{j = 1}^{j = i}/ \ \lambda_j / \ \sum_{j = 1}^{j = N_t} \lambda_j$")
ax[1].set_ylim(0,1)
ax[1].plot([r99,r99],[0,1],c = red,lw= 4)

ax[0].grid(which="both",c=gray)
ax[1].grid(which="both",c=gray)

ax[0].set_xlabel('$i$')
ax[1].set_xlabel('$i$')
plt.tight_layout()
fdir = "04_Figs/"
plt.savefig(fdir+ "eigenvalues.pdf", bbox_inches = "tight", dpi = 1000)

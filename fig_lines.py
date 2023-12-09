"""
Visualisation of Figures
@YuningW
"""

############################
# Plot Lines
###########################
print("#"*30)
print("INFO: Generating LINE Plots")
import pandas as pd
import matplotlib.pyplot as plt

plt.rc("font",family = "serif")
plt.rc("font",size = 16)
plt.rc("axes",labelsize = 18, linewidth = 2)
plt.rc("legend",fontsize= 12, handletextpad = 0.3)
plt.rc("xtick",labelsize = 18)
plt.rc("ytick",labelsize = 18)


class line_beta2:
    red = "#D23918" # luoshenzhu
    blue = "#2E59A7" # qunqing
    yellow = "#E5A84B" # huanghe liuli
    cyan = "#5DA39D" # er lv
    black = "#151D29" # lanjian
font_dict = {"weight":"bold","size":22}


mk_s = "o"; ls_s = "-"
mk_l = "v"; ls_l = "--"


fdir = "04_Figs/Lines/"
df_s = pd.read_csv("csvFile/small.csv")
df_l = pd.read_csv("csvFile/large.csv")


#------------------------------------------
# Effect of Beta

betas = [0.001,0.0025,0.005,0.01]
Num_Fields = 10000
Epochs = 300
Latent_Dim = 10


df_s_c = df_s[
            (df_s["Num_Fields"] == Num_Fields) &
            (df_s["Latent_Dim"] == Latent_Dim) &
            (df_s["Epochs"] == Epochs)]

df_l_c = df_l[
              (df_l["Num_Fields"] == Num_Fields) &
              (df_l["Latent_Dim"] == Latent_Dim) &
              (df_l["Epochs"] == Epochs)]



fig, ax = plt.subplots(1,1,sharex=True,figsize = (8,6))
ax2 = ax.twinx()

ax.tick_params("y",colors = line_beta2.blue)
ax.yaxis.label.set_color(line_beta2.blue)
ax2.tick_params("y",colors = line_beta2.red)
ax2.yaxis.label.set_color(line_beta2.red)

ax2.spines["left"].set_color(line_beta2.blue)
ax2.spines["right"].set_color(line_beta2.red)

ax.plot(df_s_c["beta"], df_s_c["Rec_Energy"],
        linestyle = ls_s, marker = mk_s,
        c= line_beta2.blue, lw = 2, markersize = 8.5)

ax.plot(df_l_c["beta"], df_l_c["Rec_Energy"],
        linestyle = ls_l, marker = mk_l,
        c= line_beta2.blue, lw = 2, markersize = 8.5)

ax2.plot(df_s_c["beta"], 100* df_s_c["Det_R"],
         linestyle = ls_s, marker = mk_s,
         c= line_beta2.red, lw = 2, markersize = 8.5)

ax2.plot(df_l_c["beta"], 100* df_l_c["Det_R"],
         linestyle = ls_l, marker = mk_l,
         c= line_beta2.red, lw = 2, markersize = 8.5)

betas_label = [r"$1\times10^{-3}$", r"$2.5\times10^{-3}$",r"$5\times10^{-3}$", r"$1\times10^{-2}$"  ]
ax.set_ylim(90, 100);
ax2.set_ylim(90, 100);
ax.set_xticklabels(betas)
ax.set_xticks(betas)
ax.set_ylabel(r"$E_k (\%)$",fontdict = font_dict)
ax.set_xlabel(r"$\beta$",fontdict = font_dict)
ax2.set_ylabel(r"${\rm det}_{\mathbf{R}}$",fontdict = font_dict)
plt.savefig(fdir+ f"Line_v2v3_n_all_m{Latent_Dim}_b_all_rec_det.pdf", bbox_inches = "tight", dpi = 1000)
print("INFO: Effect of Beta FINISH")


#------------------------------------------
# Effect of Latent Dim

beta = 0.005
Num_Fields = 10000
Epochs = 300
df_s_c = df_s[
            (df_s["Num_Fields"] == Num_Fields) &
            (df_s["beta"] == beta) &
            (df_s["Epochs"] == Epochs)
             ]
df_l_c = df_l[
            (df_l["Num_Fields"] == Num_Fields) &
            (df_l["beta"] == beta) &
            (df_l["Epochs"] == Epochs)
             ]

fig, ax = plt.subplots(1,1,sharex=True,figsize = (8,6))
ax2 = ax.twinx()



ax.tick_params("y",colors = line_beta2.blue)
ax.yaxis.label.set_color(line_beta2.blue)
ax2.tick_params("y",colors = line_beta2.red)
ax2.yaxis.label.set_color(line_beta2.red)

ax2.spines["left"].set_color(line_beta2.blue)
ax2.spines["right"].set_color(line_beta2.red)


mk_s = "o"; ls_s = "-"
mk_l = "v"; ls_l = "--"

ax.plot(df_s_c["Latent_Dim"], df_s_c["Rec_Energy"],
        linestyle = ls_s, marker = mk_s,
        c= line_beta2.blue, lw = 2, markersize = 8.5)

ax.plot(df_l_c["Latent_Dim"], df_l_c["Rec_Energy"],
        linestyle = ls_l, marker = mk_l,
        c= line_beta2.blue, lw = 2, markersize = 8.5)


ax2.plot(df_s_c["Latent_Dim"], 100* df_s_c["Det_R"],
         linestyle = ls_s, marker = mk_s,
         c= line_beta2.red, lw = 2, markersize = 8.5)

ax2.plot(df_l_c["Latent_Dim"], 100* df_l_c["Det_R"],
         linestyle = ls_l, marker = mk_l,
         c= line_beta2.red, lw = 2, markersize = 8.5)



ax.set_ylim(94, 100);
ax2.set_ylim(94, 100);
ax.set_xticklabels(df_l_c["Latent_Dim"])
ax.set_xticks(df_l_c["Latent_Dim"])
ax.set_ylabel(r"$E_k (\%)$",fontdict = font_dict )
ax.set_xlabel("$d$",fontdict = font_dict )
ax2.set_ylabel(r"${\rm det}_{\mathbf{R}}$",fontdict = font_dict )
plt.savefig(fdir+ f"Line_n_all_m_all_b_{int(beta*10000)}e-4_rec_det.pdf", bbox_inches = "tight", dpi = 1000)
print("INFO: Effect of Latent Dim FINISH")


#------------------------------------------
# Effect of Number of training data


beta = 0.005
Num_Fields = [1200,2500,5000,10000]
Epochs = 300
Latent_Dim = 10

df_s_c = df_s[
            (df_s["Latent_Dim"] == Latent_Dim) &
            (df_s["beta"] == beta) &
            (df_s["Epochs"] == Epochs)
             ]

df_l_c = df_l[
            (df_l["Latent_Dim"] == Latent_Dim) &
            (df_l["beta"] == beta) &
            (df_l["Epochs"] == Epochs)
             ]

fig, ax = plt.subplots(1,1,sharex=True,figsize = (8,6))
ax2 = ax.twinx()

ax.tick_params("y",colors = line_beta2.blue)
ax.yaxis.label.set_color(line_beta2.blue)
ax2.tick_params("y",colors = line_beta2.red)
ax2.yaxis.label.set_color(line_beta2.red)

ax2.spines["left"].set_color(line_beta2.blue)
ax2.spines["right"].set_color(line_beta2.red)

ax.plot(df_s_c["Num_Fields"], df_s_c["Rec_Energy"],
        linestyle = ls_s, marker = mk_s,
        c= line_beta2.blue, lw = 2, markersize = 8.5)

ax.plot(df_l_c["Num_Fields"], df_l_c["Rec_Energy"],
        linestyle = ls_l, marker = mk_l,
        c= line_beta2.blue, lw = 2, markersize = 8.5)

ax2.plot(df_s_c["Num_Fields"], 100* df_s_c["Det_R"],
         linestyle = ls_s, marker = mk_s,
         c= line_beta2.red, lw = 2, markersize = 8.5)

ax2.plot(df_l_c["Num_Fields"], 100* df_l_c["Det_R"],
         linestyle = ls_l, marker = mk_l,
         c= line_beta2.red, lw = 2, markersize = 8.5)



ax.set_ylim(91, 98);
ax2.set_ylim(91, 98);
ax.set_xticklabels(Num_Fields)
ax.set_xticks(Num_Fields)
ax.set_ylabel(r"$E_k (\%)$",fontdict = font_dict )
ax2.set_ylabel(r"${\rm det}_{\mathbf{R}}$",fontdict = font_dict )
ax.set_xlabel(r"$N_{\rm fields}$",fontdict = font_dict)
plt.savefig(fdir+ f"Line_n_all_m_{Latent_Dim}_b_{int(beta*10000)}e-4_rec_det.pdf", bbox_inches = "tight", dpi = 1000)

print("INFO: Effect of Nfield FINISH")
print("#"*30)



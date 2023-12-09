# Towards optimal $\beta$-variational autoencoders combined with transformers for reduced-order modeling of turbulent flows

## Introduction
+ The code in this repository features a Python implementation of reduced-order model (ROM) of turbulent flow using $\beta$-variational autoencoders and transformer neural network. More details about the implementation and results from the training are available in ["Towards optimal β-variational autoencoders combined with
transformers for reduced-order modeling of turbulent flows", Yuning Wang, Alberto Solera-Rico, Carlos Sanmiguel Vila and Ricardo Vinuesa](https://doi.org/10.1016/j.ijheatfluidflow.2023.109254)


+ We provide 

## Training and inference
### Modal decomposition: $\beta$-VAE 
+ To train $\beta$-VAE, please run:

        python vae_train.py

+ For post-processing, please run:

        python vae_postprocess.py

+ For ranking the $\beta$-VAE mode, please run:

        python vae_rankModes.py

### Temporal-dynamics prediction: Transformer / LSTM
+ To train a self-attention-based transformer, please run: 

        python temporal_pred_train_selfattn.py

+ For post-processing: 

        python temporal_pred_postprocess.py 

+ To yield the sliding-window error $\epsilon$, please run: 

        python temporal_pred_sliding_window.py 

### Archiecture
+ The transformer and LSTM archiectures are in the *utils/NNs*

+ The $\beta$-VAE archiectures are in the *utils/VAE*

+ The configurations of employed archiectures are in */utils/configs.py*

### Visualisation 
We offer the scripts and data for reproducing the figures in the paper. For instance, to visualise the results of parametric studies, please run: 

        python fig_lines.py

## Repository structure
+ **01_Data**           : The flow data of the streamwise velocity components of flow around square cylinder. Please find more details in the paper: ["Causality analysis of large-scale structures in the flow around a wall-mounted square cylinder", Álvaro Martínez-Sánchez, Esteban López, Soledad Le Clainche, Adrián Lozano-Durán, Ankit Srivastava, Ricardo Vinuesa](https://doi.org/10.1017/jfm.2023.423)

+ **02_Checkpoints**    : Save the $\beta$-VAE model, loss evolution and computation time in *.pt* format

+ **03_Mode**           : Save the temporal mode in Latent Space

+ **04_Figs**           : The figures and visualisation

+ **05_Pred**           : The data of predictions

+ **06_ROM**            : The time-series prediction for 
building the Reduce-order model (ROM)

+ **07_Loss**           : The training loss evolution during training VAE. 

+ **08_POD**            : The results from Proper-orthogonal decomposition (POD) 

+ **csvFile**          : The csv files to record $\beta$-VAE performance, where *small* denotes $Arch1$ and *large* denotes the $Arch2$, respectively. 

+ **utils**: The functions and architectures used in this project
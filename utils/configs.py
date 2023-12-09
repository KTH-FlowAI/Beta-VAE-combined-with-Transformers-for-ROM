class VAE_config:
    """
    VAE config for model trained by 10,000 snapshots
    """
    model = "v3" # v2 for small architecture and v3 for large

    beta = 0.01 # The regularisation 
    latent_dim = 10 # 
    lr = 1e-3 
    epoch = 100
    
    test_split  = 0.8 # ratio of training data before train_val_split 
    train_split = 0.8 # Ratio of val and train data during training 
    batch_size = 128
    earlystop = True
    if earlystop == True:
        patience = 30
    else:
        patience = 0

class VAE_custom:
    """
    VAE config for model trained by 26,000 snapshots

    Note that we name the Arch2 as v5 in order to tell it apart from the previous case
    """
    model       = "v5" # Name Arch1 as v4 and v5 as Arch2 v6 for an Arch between v4 and v5 

    beta        = 0.005 # The regularisation [0.001, 0.0025, 0.005, 0.01]
    latent_dim  = 10 # Latent-space dimension 
    lr          = 1e-3  # Learning Rate
    # lr          = 5e-4  # Learning Rate
    w_decay     = 0     # Weight decay if needed
    epoch       = 300 # Training Epoch

    # Kernel Size of Conv 
    knsize      = 3
    # The filter used in Encoder and decoder
    # Arch 1 [1, 16, 32, 64, 128, 256]
    # Arch 2 [1, 48, 96, 128, 256, 512] # Corrected
    if model == "v4":
        filters     = [1, 16, 32, 64, 128, 256]
    
    elif model == "v5":
        filters     = [1, 48, 96, 128, 256, 512]
    
    # Type of ConvBlock
    ## Could be useful for future work
    block_type  = "original" # "original", "bn_original", "deep", "res" "deepres"

    # Dimension of linear layer after flatten 
    linear_dim  = 128

    #activation fuction for conv layer and linear layer
    act_conv    = "elu"
    act_linear  = "elu"


    test_split  = 1 # ratio of training data before train_val_split  We use test_split = 1 
    
    if test_split == 1:
        train_split = 0.8 # Ratio of val and train data during training 
    else:
        train_split = 1 # If we do not use full dataset, all data will be used for training
    
    batch_size  = 128
    earlystop   = False
    if earlystop == True:
        patience = 50
    else:
        patience = 0



class Transformer_config:
    from utils.configs import VAE_custom

    model_type  = "TFSelf"

    in_dim      =   128 
    out_dim     =   128
    d_model     =   128
    time_proj   =   128

    next_step   =   1
    nmode       =   VAE_custom.latent_dim

    num_head    =   8
    attn_type   =   "self"

    embed       =   "time"
    num_block   =   4

    proj_dim    =   256

    act_proj    =   "relu"
    is_output   =   True
    out_act     =   None

    Epoch       =   3000
    Batch_size  =   128 # Beyond 128 then MMO
    lr          =   1e-3

    wdecay      =   0

    train_split =   0.85
    val_split   =   0.2 

    early_stop  =   True
    if early_stop:
        patience    = 30
    else:
        patience    = 0


class EasyAttn_config:
    from utils.configs import VAE_custom

    model_type  = "TFEasy"

    in_dim      =  128
    out_dim     =  128
    d_model     =  128
    time_proj   =  128

    next_step   =   1
    nmode       =   VAE_custom.latent_dim

    num_head    =   8
    attn_type   =   "easy"

    embed       =  "time"
    num_block   =   4

    proj_dim    =   256
    act_proj    =   "relu"
    is_output   =   True
    out_act     =   None

    Epoch       =   3000# 100 OR 300 OR 1000 OR 3000 
    Batch_size  =   128 # 128 or 256
    lr          =   1e-3

    wdecay      =   0

    train_split =   0.85
    val_split   =   0.2 

    early_stop  = True

    if early_stop:
        patience    = 30
    else:
        patience    = 0


class LSTM_config:
    from utils.configs import VAE_custom

    model_type  = "LSTM"

    in_dim      = 128
    d_model     = 128
    next_step   = 1
    nmode       = VAE_custom.latent_dim

    num_layer   = 4
    embed       = None
    
    hidden_size = 256

    is_output   = True
    out_act     = None

    Epoch       = 3000
    Batch_size  = 128
    lr          = 1e-3

    wdecay      = 0

    train_split = 0.85                                          
    val_split   = 0.2 
    num_train   = 135000

    early_stop =  True

    if early_stop == True:
        patience  = 30
    else:
        patience  = 0 



def Name_VAE(cfg, nt):
    """
    Name the original VAE model case
    Args:
        cfg: The class contains information of the case
        nt: Number of training data
    Returns:
        fileID: The name of the case
    """
    fileID = f"{cfg.model}_{int( nt*(cfg.train_split) )}n_{cfg.latent_dim}d_{int(cfg.beta*10000)}e-4_"+\
         f"{cfg.batch_size}bs_{cfg.epoch}epoch_{cfg.earlystop}ES_{cfg.patience}P"

    return fileID

def Name_Costum_VAE(cfg, nt):
    """
    Name the custum VAE model case
    Args:
        cfg: The class contains information of the case
        nt: Number of training data
    Returns:
        fileID: The name of the case
    """
    fileID =    f"{cfg.model}_{nt}n_{cfg.latent_dim}d_{int(cfg.beta*10000)}e-4beta_"+\
                f"{cfg.block_type}conv_{len(cfg.filters)}Nf_{cfg.filters[-1]}Fdim_{cfg.linear_dim}Ldim"+\
                f"{cfg.act_conv}convact_{cfg.act_linear}_" +\
                f"{int(cfg.lr *1e5)}e-5LR_{int(cfg.w_decay*1e5)}e-5Wd"+\
                f"{cfg.batch_size}bs_{cfg.epoch}epoch_{cfg.earlystop}ES_{cfg.patience}P"

    return fileID



def Make_Transformer_Name(cfg):
    """
    A function to name the VAE checkpoint 

    Args: 
        cfg: A class of VAE model configuration
    
    Returns:
        name: A string for Transformer model
    """

    
    case_name = f"{cfg.model_type}"+\
                f"_{cfg.in_dim}in_{cfg.d_model}dmodel_{cfg.next_step}next_{cfg.nmode}dim"+\
                f"_{cfg.embed}emb_{cfg.num_head}h_{cfg.num_block}nb_{cfg.proj_dim}ff"+\
                f"_{cfg.act_proj}act_{cfg.out_act}outact"+\
                f"_{cfg.Epoch}Epoch_{cfg.early_stop}ES_{cfg.patience}P_{int(cfg.train_split*100)}train"
    
    return case_name

def Make_LSTM_Name(cfg):
    """
    A function to name the VAE checkpoint 

    Args: 
        cfg: A class of VAE model configuration
    
    Returns:
        name: A string for LSTM model
    """
    
    case_name = f"LSTM"+\
                f"_{cfg.in_dim}in_{cfg.d_model}dmodel_{cfg.next_step}next_{cfg.nmode}dim"+\
                f"_{cfg.embed}emb_{cfg.hidden_size}hideen_{cfg.num_layer}nlayer_"+\
                f"_{cfg.out_act}outact"+\
                f"_{cfg.Epoch}Epoch_{cfg.num_train}N_{cfg.early_stop}ES_{cfg.patience}P"
    
    return case_name

class Data_config:
    time_delay  = 16 
    step = 1 
    batch_size = 64 
    n_test = 1024 
    train_shuffle = True




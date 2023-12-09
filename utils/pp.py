def l2norm_error(truth,pred,data_max):
    """
    Compute the l2-norm error for time-series prediction
    error = ||truth - pred||_2 / (2 * max(truth))

    Args:
        truth   :  The ground truth data 

        pred:   :  The predictions

        data_max:  The maximum value for groud truth data

    Returns:

        error : A NumPy array of error 
    
    """
    import numpy as np 
    import numpy.linalg as LA
    try: 
        truth.shape== pred.shape
    except: print("INFO: data not match")

    try:
        truth.shape[1] > truth.shape[0]
    except: print("The second dimension should be time")

    l2norm = LA.norm((truth - pred),axis=0)
    error = l2norm.mean()/(2*data_max) 
    return np.round(error*100,4)

def Energy_Rec(truth,pred):
    
    """
    Compute the reconstruction enery E_K using fluctuation components

    Args:
        truth   : NumPy Array with shape [N,H,W] The fluctuation componets of ground truth

        pred    : NumPy Array with shape [N,H,W] The fluctuation componets of ground pred

    Returns:

        Ek      : Energy level of reconstruction
    """
    import numpy as np 

    pred, truth = pred.squeeze(), truth.squeeze()
    err         = np.sum( (pred-truth)**2 ,axis = (0,1,2))/np.sum((truth)**2,axis = (0,1,2))
    Ek          = (1 -err) * 100 

    return Ek


def Corr_Martix(z_mean):
    """
    Compute the linear correlation between lantent variables 

    Args:
        z_mean  :  The NumPy arrays of mean variables in latent space, shape = [NTime, Nmodes] 

    Returns:     

        detR        : The determination of correlation matrix

        CorrMatrix  :  Correlation matrix in abs value in NumPy Array, shape = [Nmodes, Nmodes] 
    """
    import numpy as np 

    assert len(z_mean.shape) == 2

    Corr_Martix = abs(np.corrcoef(z_mean.T))
    detR        = np.linalg.det(Corr_Martix)

    return detR, Corr_Martix 



def Gen_SpatialMode(latent_dim,
                    model,
                    device,
                    invalue = 1,):
    """
    Generate non-linear spatial modes using decoder of VAE

    Args: 
        latent_dim     : Latent-space dimension
        model          : beta-VAE model 
        device         : The device for computation 
        invalue        : The value for modes

    Returns:
        Modes          : NumPy Array of modes, shape = [Latent, H, W]         

    """
    import torch 
    import numpy as np

    Modes = []
    
    for i in range(latent_dim):
        vector      = torch.zeros((1,latent_dim))
        vector[:,i] = invalue 
        vector      = vector.to(device).float()
        mode        = model.decoder(vector)
        Modes.append(mode.detach().cpu().numpy())
    Modes           = np.array(Modes).squeeze()

    return Modes


def Rank_SpatialMode(model, latent_dim, u_truth, u_std , modes, device):
    """
    Rank the non-linear modes in latent-space according to the energy content

    Args: 
        model           : PyTorch nn.Module as beta-VAE
        lantent_dim     : (Int) The latent-space dimension
        u_truth         : Normalised the streamwise velocity 
        u_std           : Std of the streamwise velocity 
        modes           : The reparameterized mode for decoder     
        device          : The device for running model

    Returns:
        Ranks           : The ranks for non-linear modes 
        Ecum            : Ecumlative energy for each rank
    """
    

    import numpy as np 
    import torch 
    from torch.utils.data import DataLoader, TensorDataset

    print(f"The modes has shape of {modes.shape}")
    u_truth = u_truth * u_std
    Ranks = np.zeros(latent_dim, dtype=int)
    Latent_Range = np.arange(latent_dim)

    Ecum = []
    partialModes = np.zeros_like(modes, dtype=np.float32)

    for i in range(latent_dim):
        Eks = []
        print(f"\nAt element {i}:\n")
        for j in Latent_Range:  # for mode in remaining modes
            print(Ranks[:i], j, end="")
            partialModes *= 0
            partialModes[:, Ranks[:i]] = modes[:, Ranks[:i]]
            partialModes[:, j] = modes[:, j]
            u_pred = []

            dl = DataLoader(
                            TensorDataset(torch.from_numpy(partialModes)),
                            batch_size= 64,
                            )

            for pmode in range(partialModes.shape[0]):
                
                u_p = model.decoder(torch.from_numpy(partialModes[np.newaxis,pmode,:]).float().to(device))
                u_pred.append(u_p.detach().cpu().numpy())
            
            u_pred = np.array(u_pred).squeeze()

            u_pred *= u_std
            try:
                u_pred.shape == u_truth.shape 
            except: 
                print(f"The shape Not matcheed")
                quit()

            energy = Energy_Rec(u_truth, u_pred)
            Eks.append(energy)
            
            del u_pred
            print(f'For mode {j}: Ek={Eks[-1]}')
        Eks = np.array(Eks).squeeze()
        ind = Latent_Range[np.argmax(Eks)]
        Ranks[i] = ind
        Latent_Range = np.delete(Latent_Range, np.argmax(Eks))
        Ecum.append(np.max(Eks))
        print('Adding: ', ind, ', Ek: ', np.max(Eks))
        print('############################################')
    
    Ecum = np.array(Ecum)
    print(f"Rank finished, the rank is {Ranks}")
    print(f"Cumulative Ek is {Ecum}")

    return np.array(Ranks), Ecum




def Sliding_Window_Error(test_data,
                         model, device,
                         in_dim,window = 100):
    """
    Compute the sliding window error on test dataset
    Args:
        test_data   : A numpy array of test data [Ntime, Nmode]
        model       : A torch.nn.Module as model 
        device      : String of name of device
        in_dim      : Integar of input dimension
        window      : The size of window for evaluation, default = 100 
    
    Returns:
        error_l2    : A numpy arrary of sliding window error, shape = [window,]
    
    """
    import torch
    import copy
    from tqdm import tqdm
    import numpy as np 

    def l2norm(predictions, targets):
        return  np.mean(np.sqrt( (predictions - targets) ** 2 ), axis=1 ) 
    model.to(device)
    model.eval()

    SeqLen = test_data.shape[0]
    error = None
    for init in tqdm(range(in_dim,SeqLen-window, 5)):
        temporalModes_pred = copy.deepcopy(test_data)

        for timestep in range(init, init+window):

            data    = temporalModes_pred[None, (timestep-in_dim):timestep, :]
            feature = torch.from_numpy(data).float().to(device)
            pred    = model(feature)
            temporalModes_pred[timestep,:] = pred[0].detach().cpu().numpy()

        if error is None:
            error = l2norm(temporalModes_pred[init:init+window,:], test_data[init:init+window,:])
            n = 1
        else:
            error = error + l2norm(temporalModes_pred[init:init+window,:], test_data[init:init+window,:])
            n += 1

    print(n)
    error_l2 = error / n 
    print(error.shape)

    return error_l2

def make_Prediction(test_data, model,
                    device, in_dim, next_step):
    """
    Function for generat the prediction data 
    
    Args:
        test_data   :  A numpy array of test data, with shape of [Ntime, Nmode]
        model       :  A torch.nn.Module object as model
        device      :  String of name of device    
        in_dim      :  Integar of TimeDelay size
        next_step   :  Future time step to predict
    
    Returns:
        preds    : A numpy array of the prediction  
    """
    from copy import deepcopy
    import torch 
    from tqdm import tqdm


    
    model.eval()
    model.to(device)
    Preds  = deepcopy(test_data)
    seq_len = max([Preds.shape[0],Preds.shape[1]])
    print(f"The sequence length = {seq_len}")

    for i in tqdm(range(in_dim,seq_len-next_step)):
        
        feature = Preds[None,i-in_dim:i,:]

        x = torch.from_numpy(feature)
        x = x.float().to(device)
        pred = model(x)

        pred = pred.cpu().detach().numpy()

        Preds[i:i+next_step,:] = pred[0,:,:]

    return Preds



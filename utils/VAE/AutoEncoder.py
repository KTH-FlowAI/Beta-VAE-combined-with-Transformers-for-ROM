"""
Class for beta-AutoEncoder 
"""
import torch 
from torch import nn 
from utils.VAE.EnDecoder import *


class BetaVAE(nn.Module):
    """
    nn.Module for Beta-VAE architecture and models 
    Args:  
        zdim        :   The dimension of latent space
        knsize      :   The kernel size for Conv Layer
        beta        :   The value of regularisation term beta for kl-divergence
        knsize      : kernel size of Convolution layer
        filters     : A list of number of filters used in Transpose Convblock
        block_type  : The type of ConvBlock used in architecture 
        lineardim   : The dimension of linear layer 
        act_conv    : The activation fuction for convblock
        act_linear  : The activation fuction for linear layer 
    
    func: 
        rec_loss()        : Compute the reconstruction loss via MSE
        kl_loss()         : Compute the kl-divergence loss 
        vae_loss()        : Compute the total loss = E_rec + beta * Kl-div
        reparameterize()  : Implement reparameterisation trick
        forward()         : Forward propergation of the model

            
    """    
    def __init__(self,
                    zdim,knsize,beta,
                    filters             = [256,128,64,32,16,1],
                    block_type          = "deep",
                    lineardim           = 128, 
                    act_conv            = "elu",
                    act_linear          = "elu"
                 ) -> None:
        super(BetaVAE,self).__init__()
        self.zdim       = zdim
        self.beta       = beta
        self.encoder    = encoder(zdim,knsize,
                               filters      =   filters,
                               block_type   =   block_type, 
                               lineardim    =   lineardim,
                               act_conv     =   act_conv,
                               act_linear   =   act_linear)
        
        comp_shape = self.encoder.compute_compression(filters=filters[1:])
        print(f"The size after compression is {comp_shape}")
        filters.reverse()
        self.decoder = decoder(zdim,knsize,
                               compress_shape =   comp_shape,
                               filters        =   filters,
                               block_type     =   block_type,
                               lineardim      =   lineardim,
                               act_conv       =   act_conv,
                               act_linear     =   act_linear)
    
        self.mse = nn.MSELoss()
    def rec_loss(self,pred,y):
        loss = self.mse(pred,y)
        return loss
    
    def kl_loss(self,z_mean,z_log_var):
        kl_loss = 1 + z_log_var - torch.square(z_mean)-torch.exp(z_log_var)
        kl_loss *= -0.5
        return torch.mean(kl_loss)
    
    def vae_loss(self,rec_loss,kl_loss):
        loss = rec_loss + self.beta * kl_loss
        return torch.mean(loss)
    
    def reparameterize(self,args):
        """
        Reparameterisation in latent space:

        Args:
            args       :  A tuple of (z_mean, z_var)
        
        """
        z_mean,z_log_sigma = args
        epsilon = torch.randn_like(z_log_sigma)
        return z_mean +  torch.exp(0.5*z_log_sigma) * epsilon
     
    def forward(self,x):
        z_mean,z_var = self.encoder(x)
        z_out = self.reparameterize((z_mean,z_var))
        out = self.decoder(z_out)
        return z_mean,z_var, out





if __name__ == "__main__":
    # filters = [1,16, 32, 64, 128,256]
    filters = [1 ,48, 96, 128 , 256 , 512]
    z_dim  = 10 
    knsize = 3 
    act_conv = 'tanh'
    act_linear = 'tanh'
    
    x = torch.randn(size=(1, 1, 96, 192)).float()
    
    model = BetaVAE(zdim= z_dim, 
                    knsize= knsize, 
                    beta= 1e-3, 
                    filters= filters,
                    lineardim= 128,
                    act_conv=act_conv,
                    act_linear= act_linear)
    try:
        y = model(x)
        print(f"INFO: The forward prop has been valid")
    except:
        print(f"The model archiecture when wrong")
        quit()

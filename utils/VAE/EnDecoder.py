"""
Architecture of VAE
"""
import torch 
from torch import nn 
import torch.nn.functional as F
import numpy as np 
from utils.VAE.ConvBlock import *

###########################################
## Encoder Module
###########################################
class encoder(nn.Module):
    def __init__(self,
                    zdim,
                    knsize,
                    filters     = [1,16,32,64,128,256],
                    block_type  = "deep",
                    lineardim    = 128, 
                    act_conv    = "elu",
                    act_linear   = "elu") -> None:
        """
        Module of encoder architecture
        
        Args:
            zdim:   latent_dim
            knsize: kernel size of Convolution layer
            filters: A list of number of filters used in Convblock
            block_type: The type of ConvBlock used in architecture 
            lineardim: The dimension of linear layer 
            act_conv : The activation fuction for convblock
            act_linear : The activation fuction for linear layer

        
        """
        super(encoder,self).__init__()
        self.knsize = knsize
        self.zdim = zdim

        self.h, self.w, self.c = 96, 192, 1
        
        c_comp, h_comp, w_comp= self.compute_compression(filters[1:])   

        if (int(h_comp) == h_comp) and (int(w_comp) == w_comp):
            print(f"The stragtegy of compression is valid, the final size of domain is [{c_comp, h_comp, w_comp}], flatten = ({c_comp * h_comp * w_comp})")
            
        else:
            print("The compression does not work!!")        

        self.ConvBlocks = nn.Sequential()
        for i, f in enumerate(filters[:-1]): 
            if block_type  == "original":
                self.ConvBlocks.add_module(
                                        f"Down{i}",DownBlock(   in_channel  =   filters[i], 
                                                                out_channel =   filters[i+1],
                                                                knsize      =   knsize,
                                                                activation  =   act_conv)
                                        )
            
            if block_type  == "bn_original":
                self.ConvBlocks.add_module(
                                        f"Down{i}",BNDownBlock(   in_channel  =   filters[i], 
                                                                out_channel =   filters[i+1],
                                                                knsize      =   knsize,
                                                                activation  =   act_conv)
                                        )
            
            if block_type  == "deep":
                self.ConvBlocks.add_module(
                                        f"Down{i}",deepDownBlock(   in_channel  =   filters[i], 
                                                                out_channel =   filters[i+1],
                                                                knsize      =   knsize,
                                                                activation  =   act_conv)
                                        )
            if block_type  == "res":
                self.ConvBlocks.add_module(
                                        f"Down{i}",ResDownBlock(   in_channel  =   filters[i], 
                                                                out_channel =   filters[i+1],
                                                                knsize      =   knsize,
                                                                activation  =   act_conv)
                                        )
                
        
        
        self.flat  = nn.Flatten()

        
        self.linear = nn.Linear(in_features= (int(h_comp) * int(w_comp) * int(c_comp)),
                                out_features= lineardim)
        
        if act_linear =="elu":
            self.act = nn.ELU()
        if act_linear == "tanh":
            self.act = nn.Tanh()

        self.lin_mean = nn.Linear(in_features=lineardim,out_features=zdim)
        self.lin_var = nn.Linear(in_features=lineardim,out_features=zdim)
        
        nn.init.xavier_uniform_(self.lin_mean.weight)
        nn.init.xavier_uniform_(self.lin_var.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.lin_var.bias)
        nn.init.zeros_(self.lin_mean.bias)
        nn.init.zeros_(self.linear.bias)
    
    def compute_compression(self, filters:list):
        """
        Compute the compression size of the input domain 
        Args: 
            self: Gives the channel, height and weight of domain 
            filers: List of filter number 
        
        Returns:
            (c_comp, h_comp, w_comp): [C, H, W] The size of domain after compression
        """ 
        Nostep  = len(filters) 
        h_comp  = self.h * 0.5 ** Nostep
        w_comp  = self.w * 0.5 ** Nostep
        c_comp  = filters[-1]   

        return (c_comp, h_comp, w_comp) 


    
    def forward(self,x):
        x = self.ConvBlocks(x)
        x = self.flat(x)
        x = self.act(self.linear(x))
        z_mean = self.lin_mean(x)
        z_var = self.lin_var(x)

        return (z_mean,z_var)


###########################################
## Decoder Module
###########################################
class decoder(nn.Module):
    def __init__(   self,
                    zdim,
                    knsize,
                    compress_shape  = (256,3,6),
                    block_type      = "deep",
                    filters         = [256,128,64,32,16,1],
                    lineardim        = 128, 
                    act_conv        = "elu",
                    act_linear      = "elu"
                    ) -> None:
        """
        Module of decoder architecture
        
        Args:
            zdim            : latent-space dimension
            knsize          : kernel size of Convolution layer
            compress_shape  : The shape of the compressed domain
            filters         : A list of number of filters used in Transpose Convblock
            block_type      : The type of ConvBlock used in architecture 
            lineardim       : The dimension of linear layer 
            act_conv        : The activation fuction for convblock
            act_linear      : The activation fuction for linear layer

        
        """
        super(decoder,self).__init__()

        print(f"Reverse filter is {filters}")
        self.zdim = zdim
        self.comp_c, self.comp_h, self.comp_w = compress_shape
        
        self.linear = nn.Linear(self.zdim,lineardim)
        self.recover = nn.Linear(lineardim,int(self.comp_c) * int(self.comp_h) * int(self.comp_w))
        
        if act_linear == "elu":
            self.act = nn.ELU()
        if act_linear == 'tanh':
            self.act = nn.Tanh()

        self.TransConvBlocks = nn.Sequential()
        for i,f in enumerate(filters[:-2]):
            if block_type  == "original":
                self.TransConvBlocks.add_module(
                                        f"Up{i}",UpBlock(   in_channel  =   filters[i], 
                                                                out_channel =   filters[i+1],
                                                                knsize      =   knsize,
                                                                activation  =   act_conv)
                                        )
            if block_type  == "bn_original":
                self.TransConvBlocks.add_module(
                                        f"Up{i}",BNUpBlock(   in_channel  =   filters[i], 
                                                                out_channel =   filters[i+1],
                                                                knsize      =   knsize,
                                                                activation  =   act_conv)
                                        )
            
            
            
            if block_type  == "deep":
                self.TransConvBlocks.add_module(
                                        f"Up{i}",deepUpBlock(   in_channel  =   filters[i], 
                                                                out_channel =   filters[i+1],
                                                                knsize      =   knsize,
                                                                activation  =   act_conv)
                                        )
            if block_type  == "res":
                self.TransConvBlocks.add_module(
                                        f"Up{i}",ResUpBlock(   in_channel  =   filters[i], 
                                                                out_channel =   filters[i+1],
                                                                knsize      =   knsize,
                                                                activation  =   act_conv)
                                        )

        self.conv = nn.ConvTranspose2d(filters[-2],filters[-1],
                                       knsize,stride=2,
                                       padding=1,output_padding=1)
        
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

        nn.init.xavier_normal_(self.linear.weight)        
        nn.init.xavier_normal_(self.recover.weight)
        nn.init.zeros_(self.linear.bias)        
        nn.init.zeros_(self.recover.bias)        

    def forward(self,x):
        x = self.act(self.linear(x))
        x = self.act(self.recover(x))
        
        x = x.reshape(x.size(0),
                      int(self.comp_c),int(self.comp_h),int(self.comp_w))

        x = self.TransConvBlocks(x)
        x = self.conv(x)
        return x




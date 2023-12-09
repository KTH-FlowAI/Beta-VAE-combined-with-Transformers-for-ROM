"""
A script for Various of convolution block
"""
import torch 
from torch import nn as nn
import torch.nn.functional as F 

############################
# Origin Conv Block
###########################

class DownBlock(nn.Module):
    def __init__(self,in_channel,out_channel,
                        knsize,
                        activation="elu") -> None:
        """
        The Original DownSampler block used in Encoder 
        Args:
            in_channel:  The number of input channel 
            out_channel: The number of output channel 
            knsize:      Size of filter 
            activation:  Activation used in forward
        """
        
        super(DownBlock,self).__init__()
        self.conv     = nn.Conv2d(in_channel,out_channel,kernel_size=knsize,stride=2,padding=1)
        if activation == "elu":
            self.act  = nn.ELU()
        if activation == "tanh":
            self.act  = nn.Tanh()
        if activation == "swish":
            self.act  = nn.SiLU()
        
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        print("conv has been initialized")
    def forward(self,x):
        return  self.act((self.conv(x)))

class UpBlock(nn.Module):
    """
    The Original UpSampler block used in Decoder 
    """
    
    def __init__(self,in_channel,out_channel,
                        knsize,
                        activation='elu') -> None:
        """
        The Original UpSampler block used in decoder 
        Args:
            in_channel:  The number of input channel 
            out_channel: The number of output channel 
            knsize:      Size of filter 
            activation:  Activation used in forward
        """
        
        super(UpBlock,self).__init__()
        self.conv   = nn.ConvTranspose2d(in_channel,out_channel,knsize,stride = 2,output_padding=1,padding=1)
        
        if activation == "elu":
            self.act = nn.ELU()
        if activation == 'tanh':
            self.act = nn.Tanh()
        if activation == "swish":
            self.act  = nn.SiLU()
        

        nn.init.xavier_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        print("conv has been initialized")
    def forward(self,x):
        x = self.conv(x)
        x = self.act(x)
        return x 

#########################
# Conv Block with Batch Normalisation
########################
class BNDownBlock(nn.Module):
    def __init__(self,in_channel,out_channel,
                        knsize,
                        activation="elu") -> None:
        """
        The DownSampler block used in Encoder 
        Args:
            in_channel:  The number of input channel 
            out_channel: The number of output channel 
            knsize:      Size of filter 
            activation:  Activation used in forward
        """
        
        super(BNDownBlock,self).__init__()
        
        conv1     = nn.Conv2d(in_channel,out_channel,kernel_size=knsize,stride=2,padding=1)
        
        nn.init.xavier_normal_(conv1.weight)
        nn.init.zeros_(conv1.bias)
        
    
        
        if activation == "elu":
            self.act  = nn.ELU()
        if activation == "tanh":
            self.act  = nn.Tanh()
        if activation == "swish":
            self.act  = nn.SiLU()
        
        self.conv1 = nn.Sequential(
                                    conv1,
                                   nn.BatchNorm2d(num_features= out_channel,eps= 1e-3),
                                   self.act,
                                   )    
        print("conv has been initialized")
    
    
    def forward(self,x):
        x = self.conv1(x)
        return x


class BNUpBlock(nn.Module):
    def __init__(self,in_channel,out_channel,
                        knsize,
                        activation="elu") -> None:
        """
        The Deeper UpSampler block used in decoder 
        Args:
            in_channel:  The number of input channel 
            out_channel: The number of output channel 
            knsize:      Size of filter 
            activation:  Activation used in forward
        """
        
        super(BNUpBlock,self).__init__()
        
        conv1   = nn.ConvTranspose2d(in_channel,out_channel,knsize,stride = 2,output_padding=1,padding=1)
      
        nn.init.xavier_normal_(conv1.weight)
        nn.init.zeros_(conv1.bias)
    
    
        
        if activation == "elu":
            self.act  = nn.ELU()
        if activation == "tanh":
            self.act  = nn.Tanh()
        if activation == "swish":
            self.act  = nn.SiLU()
        
        self.conv1 = nn.Sequential(
                                    conv1,
                                   nn.BatchNorm2d(num_features= out_channel,eps= 1e-3),
                                   self.act,
                                   )
        
        print("conv has been initialized")
    def forward(self,x):
        x = self.conv1(x)
        return x 



############################
# deep Conv Block
###########################


class deepDownBlock(nn.Module):
    def __init__(self,in_channel,out_channel,
                        knsize,
                        activation="elu") -> None:
        """
        The Deeper DownSampler block used in Encoder 
        Args:
            in_channel:  The number of input channel 
            out_channel: The number of output channel 
            knsize:      Size of filter 
            activation:  Activation used in forward
        """
        
        super(deepDownBlock,self).__init__()
        
        conv1     = nn.Conv2d(in_channel,out_channel,kernel_size=knsize,stride=2,padding=1)

        conv2     = nn.Conv2d(out_channel,out_channel,kernel_size= 1,stride=1,padding=0)
        
        nn.init.xavier_normal_(conv1.weight)
        nn.init.xavier_normal_(conv2.weight)
        nn.init.zeros_(conv1.bias)
        nn.init.zeros_(conv2.bias)

    
        
        if activation == "elu":
            self.act  = nn.ELU()
        if activation == "tanh":
            self.act  = nn.Tanh()
        if activation == "swish":
            self.act  = nn.SiLU()
        
        self.conv1 = nn.Sequential(
                                    conv1,
                                #    nn.BatchNorm2d(num_features= out_channel,eps= 1e-3),
                                   self.act,
                                   )
        
        self.conv2 = nn.Sequential(
                                    conv2, 
                                    # nn.BatchNorm2d(num_features= out_channel,eps= 1e-3),
                                    # self.act,
        )
            
        print("conv has been initialized")
    
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        # return self.act(x)
        return x


class deepUpBlock(nn.Module):
    def __init__(self,in_channel,out_channel,
                        knsize,
                        activation="elu") -> None:
        """
        The Deeper UpSampler block used in decoder 
        Args:
            in_channel:  The number of input channel 
            out_channel: The number of output channel 
            knsize:      Size of filter 
            activation:  Activation used in forward
        """
        
        super(deepUpBlock,self).__init__()
        
        conv1   = nn.ConvTranspose2d(in_channel,out_channel,knsize,stride = 2,output_padding=1,padding=1)
        
        conv2   = nn.Conv2d(out_channel,out_channel,kernel_size= 1,stride=1,padding=0)
        
        nn.init.xavier_normal_(conv1.weight)
        nn.init.xavier_normal_(conv2.weight)
        nn.init.zeros_(conv1.bias)
        nn.init.zeros_(conv2.bias)

    
        
        if activation == "elu":
            self.act  = nn.ELU()
        if activation == "tanh":
            self.act  = nn.Tanh()
        if activation == "swish":
            self.act  = nn.SiLU()
        
        self.conv1 = nn.Sequential(
                                    conv1,
                                #    nn.BatchNorm2d(num_features= out_channel,eps= 1e-3),
                                   self.act,
                                   )
        
        self.conv2 = nn.Sequential(
                                    conv2, 
                                    # nn.BatchNorm2d(num_features= out_channel,eps= 1e-3),
                                    # self.act,
        )
        
        print("conv has been initialized")
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        # return self.act(x)
        return self.act(x)





############################
# Residual Conv Block
###########################
class ResDownBlock(nn.Module):
    def __init__(self,in_channel,out_channel,
                        knsize,
                        activation="elu") -> None:
        """
        The Residual DownSampler block used in Encoder 
        Args:
            in_channel:  The number of input channel 
            out_channel: The number of output channel 
            knsize:      Size of filter 
            activation:  Activation used in forward
        """
        
        super(ResDownBlock,self).__init__()
        
        conv1     = nn.Conv2d(in_channel,out_channel,kernel_size=knsize,stride=2,padding=1)

        conv2     = nn.Conv2d(in_channel,out_channel,kernel_size=knsize,stride=2,padding=1)
        
        
        nn.init.xavier_normal_(conv1.weight)
        nn.init.xavier_normal_(conv2.weight)
        
        nn.init.zeros_(conv1.bias)
        nn.init.zeros_(conv2.bias)
        
        if activation == "elu":
            act  = nn.ELU()
        if activation == "tanh":
            act  = nn.Tanh()
        if activation == "swish":
            act  = nn.SiLU()

        self.conv1 = nn.Sequential(
                        conv1, 
                        nn.BatchNorm2d(out_channel),
                        act,) 

        self.conv2 = nn.Sequential(
                        conv2,
                        )       

        self.act   = act 
        print("conv has been initialized")
    def forward(self,x):
        # Residual
        x_ = self.conv2(x)
        # Forward
        x = self.conv1(x)
        return x_ + x 
    
class ResUpBlock(nn.Module):
    def __init__(self,in_channel,out_channel,
                        knsize,
                        activation="elu") -> None:
        """
        The Residual UpSampler block used in Decoder 
        Args:
            in_channel:  The number of input channel 
            out_channel: The number of output channel 
            knsize:      Size of filter 
            activation:  Activation used in forward
        """
        
        super(ResUpBlock,self).__init__()
        
        conv1     = nn.ConvTranspose2d(in_channel,out_channel,kernel_size=knsize,stride=2,padding=1,output_padding=1)

        conv2     = nn.ConvTranspose2d(in_channel,out_channel,kernel_size=knsize,stride=2,padding=1,output_padding=1)
        
        
        
        nn.init.xavier_normal_(conv1.weight)
        nn.init.xavier_normal_(conv2.weight)
        nn.init.zeros_(conv1.bias)
        nn.init.zeros_(conv2.bias)
        
        if activation == "elu":
            act  = nn.ELU()
        if activation == "tanh":
            act  = nn.Tanh()
        if activation == "swish":
            act  = nn.SiLU()

        self.conv1 = nn.Sequential(
                        conv1, 
                        nn.BatchNorm2d(out_channel),
                        act, 
                    
        )        

        self.conv2 = nn.Sequential(
                        conv2,
        )
        
        self.act   = act


        print("conv has been initialized")
    def forward(self,x):
        # Residual
        x_ = self.conv2(x)
        # Forward
        x = self.conv1(x)
        return x_ + x 
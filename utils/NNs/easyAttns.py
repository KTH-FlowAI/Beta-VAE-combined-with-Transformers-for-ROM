"""
Use the easy-attention mechanism for time-series prediction in latent space 
We use a stack of encoders only for the prediction, leveraging time-delay embedding 

@author yuningw
"""


import  torch 
from    torch               import nn 
import  torch.nn.functional as      F
from    utils.NNs.layers    import PositionWiseFeedForward
from    utils.NNs.Embedding import *


class DenseEasyAttn(nn.Module):
    
    def __init__(self,  d_model, 
                        seqLen,
                        num_head,) -> None:
        """
        Dense Easy attention mechansim used in transformer model for the time-series prediction and reconstruction
        
        Args:

            d_model     :   The embedding dimension for the input tensor 
            
            seqLen      :   The length of the sequence 

            num_head    :   The number of head to be used for multi-head attention
    
        """
        super(DenseEasyAttn,self).__init__()
     
        assert d_model % num_head == 0, "dmodel must be divible by number of heads"

        self.d_model    =   d_model
        self.d_k        =   d_model // num_head
        self.num_heads  =   num_head
        # Create the tensors
        self.Alpha      = nn.Parameter(torch.randn(size=(num_head,seqLen,seqLen),dtype=torch.float),requires_grad=True)            
        self.WV         = nn.Parameter(torch.randn(size=(seqLen,d_model)       ,dtype=torch.float),requires_grad=True)               
        # Initialisation
        nn.init.xavier_uniform_(self.Alpha)
        nn.init.xavier_uniform_(self.WV)
    
    def split_heads(self, x):
        """
        Split the sequence into multi-heads 

        Args:
            x   : Input sequence shape = [B, S, N]
        
        Returns:
            x   : sequence with shape = [B, H, S, N//H]
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        """
        Combine the sequence into multi-heads 

        Args:
            x   : Input sequence shape = [B, H, S, N//H]
        
        Returns:
            x   : sequence with shape = [B, S, N]
        """
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
 
    def forward(self,x:torch.Tensor):   
        """
        Forward prop for the easyattention module 
        
        Following the expression:  x_hat    =   Alpha @ Wv @ x 

        Args:  
            
            self    :   The self objects

            x       :   A tensor of Input data
        
        Returns:
            
            x       :   The tensor be encoded by the moudle
        
        """
        # Obtain the value of batch size 
        B,_,_   =   x.shape
        # We repeat the tensor into same number of batch size that input has 
        Wv      =   self.WV.repeat(B,1,1)    
        # Implement matmul along the batch 
        V       =   torch.bmm(Wv,x)
        # Split heads for value
        V_h     =   self.split_heads(V)
        # Implement the learnable attention tensor 
        Alpha   =   self.Alpha.repeat(B,1,1,1)
        # Gain Attention by matrix multiplication
        x       =   Alpha @ V_h
        # Combine the output back to the original size
        x       =   self.combine_heads(x)
        
        return x



class easyEncoderLayer(nn.Module):
    def __init__(self, d_model, seqLen, num_heads, d_ff, dropout, act_proj):
        """
        nn.Module for transformer Encoder layer
        
        Args:
            d_model     :   (Int) The embedding dimension 
            
            seqLen      :   (Int) The length of the input sequence
            
            num_heads   :   (Int) The number of heads used in attention module

            
            d_ff        :   (Int) Projection dimension used in Feed-Forward network 
            
            dropout     :   (Float) The dropout value to prevent from pverfitting

            act_proj    :   (Str)   The activation function used in the FFD
        """
        super(easyEncoderLayer, self).__init__()
        self.attn = DenseEasyAttn(d_model, seqLen, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff,act_proj)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        The forward prop for the module 
        Args:
            x       :   Input sequence 
            
        Returns:
            x       :   The encoded sequence in latent space       
        """
        attn_output = self.attn(x)

        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x




class easyTransformerEncoder(nn.Module):
    """
    A transformer-based architecture using temporal-spatial embedding and a stack of encoder 


    """

    def __init__(self, d_input, d_output, seqLen, 
                        d_proj, d_model, d_ff,
                         num_head,num_layer,
                        act_proj= "relu",
                        dropout= 1e-5) -> None:
        super(easyTransformerEncoder,self).__init__()

        self.embed      =   TimeSpaceEmbedding(d_input, seqLen, d_proj, d_model)

        self.encoders   =   nn.ModuleList([ easyEncoderLayer(d_model    =   d_model, 
                                                             seqLen     =   d_proj,
                                                             num_heads  =   num_head ,
                                                             d_ff       =   d_ff,
                                                             act_proj   =   act_proj,
                                                             dropout    =   dropout ) for _ in range(num_layer)]) 

        self.cf         =   nn.Conv1d(d_proj, d_output,1)
        
        self.of         =   nn.Linear(d_model,seqLen)


        nn.init.xavier_uniform_(self.cf.weight)
        nn.init.xavier_uniform_(self.of.weight)
        nn.init.zeros_(self.cf.bias)
        nn.init.zeros_(self.of.bias)

    def forward(self, src):
        enc_input   = self.embed(src)
        # print(enc_input.shape)
        # Leave the residual for forward porp

        for enc_layer in self.encoders:
            
            enc_input   = enc_layer(enc_input)


        x   =   self.cf(enc_input)
        # print(x.shape)
        x   =   self.of(x)

        return x 







if __name__ == "__main__":
    # An example for using easy attention
    d_model     =   64 
    seqLen      =   64 
    num_head    =   4
    Batch_size  =   16

    d_input     =   64 
    d_output    =   1
    d_ff        =   96 
    d_proj      =   72
    num_layer   =   2
    

    # Give a tensor 
    x           =   torch.randn(size=(Batch_size,d_input,d_model))

    # Create a module
    eyattn      =   easyTransformerEncoder(
                                            d_input     =   d_input,
                                            d_output    =   d_output,
                                            seqLen      =   seqLen,
                                            d_proj      =   d_proj,
                                            d_model     =   d_model,
                                            d_ff        =   d_ff,
                                            num_head    =   num_head,
                                            num_layer   =   num_layer

                                                )

    # Output 
    p           =   eyattn(x.float())

    print(f"Suppose we have a tensor {x.shape} as Input")
    print(f"The output by easy attention is  {p.shape}")
"""
Script for the encoder and decoder layer for full transformer
"""

from    utils.NNs.attns import MultiHeadAttention
from    torch           import nn 

class PositionWiseFeedForward(nn.Module):
    def __init__(self, 
                 d_model, 
                 d_ff, 
                 activation="relu"):
        """
        The nn.Module for Feed-Forward network in transformer encoder/decoder layer 
        
        Args:
            d_model     :  (Int) The dimension of embedding 

            d_ff        :  (Int) The projection dimension in the FFD 
            
            activation  :  (Str) Activation function used in network

        """
        
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        
        if activation == "relu":
            self.act = nn.ReLU()
        if activation == "gelu":
            self.act = nn.GELU()
        if activation == "elu":
            self.act = nn.ELU()


    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))



class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, act_proj):
        """
        nn.Module for transformer Encoder layer
        
        Args:
            d_model     :   (Int) The embedding dimension 
            
            num_heads   :   (Int) The number of heads used in attention module
            
            d_ff        :   (Int) Projection dimension used in Feed-Forward network 
            
            dropout     :   (Float) The dropout value to prevent from pverfitting

            act_proj    :   (Str)   The activation function used in the FFD
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff,act_proj)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        """
        The forward prop for the module 
        Args:
            x       :   Input sequence 
            
            mask    :   the mask used for attention, usually be the src_mask

        Returns:
            x       :   The encoded sequence in latent space       
        """
        attn_output = self.self_attn(x, x, x, mask)

        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, act_proj):
        """
        nn.Module for transformer Decoder layer
        
        Args:
            d_model     :   (Int) The embedding dimension 
            
            num_heads   :   (Int) The number of heads used in attention module
            
            d_ff        :   (Int) Projection dimension used in Feed-Forward network 
            
            dropout     :   (Float) The dropout value to prevent from pverfitting

            act_proj    :   (Str)   The type of activation in FFD 
        """
        super(DecoderLayer, self).__init__()
        # Attention for input sequence of decoder 
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # Attention for the encoder output and the decoder 
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, act_proj)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        """
        The forward prop for the module 
        Args:
            x           :   Input sequence 
            
            src_mask    :   Source mask for encoder output 
            
            tgt_mask    :   Target mask for model from "cheating" 

        Returns:
            x       :   The output sequence from decoder       
        """
        
        attn_output = self.self_attn(x, x, x, tgt_mask)

        x = self.norm1(x + self.dropout(attn_output))

        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)

        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    
"""
The transfomer encoders using new embeding layer

"""

from utils.NNs.layers           import EncoderLayer, DecoderLayer
from utils.NNs.Embedding        import TimeSpaceEmbedding
import  torch
from    torch                   import nn

class EmbedTransformerEncoder(nn.Module):
    """
    A transformer-based architecture using temporal-spatial embedding and a stack of encoder 


    """

    def __init__(self, d_input, d_output, n_mode, 
                        d_proj, d_model, d_ff,
                         num_head,num_layer,
                        act_proj= "relu",
                        dropout= 1e-5) -> None:
        super(EmbedTransformerEncoder,self).__init__()

        self.embed      =   TimeSpaceEmbedding(d_input, n_mode, d_proj, d_model)

        self.encoders   =   nn.ModuleList([ EncoderLayer(d_model= d_model, num_heads=num_head ,d_ff = d_ff,act_proj=act_proj,dropout=dropout ) for _ in range(num_layer)]) 

        self.cf         =   nn.Conv1d(d_proj, d_output,1)
        
        self.of         =   nn.Linear(d_model,n_mode)

        nn.init.xavier_uniform_(self.cf.weight)
        nn.init.xavier_uniform_(self.of.weight)
        nn.init.zeros_(self.cf.bias)
        nn.init.zeros_(self.of.bias)

    def forward(self, src):
        enc_input   = self.embed(src)
        # Leave the residual for forward porp
        enc_res     = 0
        for enc_layer in self.encoders:
            enc_input   = enc_layer(enc_input+enc_res,None)  
            enc_res     = enc_input  

        x   =   self.cf(enc_input)
        x   =   self.of(x)

        return x 


if __name__ == "__main__":

    d_input = 64
    d_output = 1
    n_mode = 20
    d_proj = 128
    d_model = 64
    d_ff    = 128
    batch = 10
    print(f"Suppose we have input with size of [{batch}, {d_input}, {n_mode}]")

    print(f"We need output with size of [{batch}, {d_proj},{d_model}]")

    model = EmbedTransformerEncoder( 
                                    d_input = d_input,
                                    d_output= d_output,
                                    n_mode  = n_mode,
                                    d_proj  = d_proj,
                                    d_model = d_model,
                                    d_ff    = d_ff,
                                    num_layer = 2,
                                    
                                        ) 




    # give an input 

    x = torch.randn(size=(batch,d_input,n_mode))

    print(f"We generate a input with size of {x.shape}")

    p = model(x.float())

    print(f"The forward success, shape of output = {p.shape}")

import torch
import torch.nn as nn

"""
TGAT layer 
input

"""

class TGATLayer(nn.Module):
    def __init__(self,
        node_dim:int,
        latent_dim:int=32,
        num_heads:int=1,
        is_last:bool=True,
        **kwargs
        ):
        super().__init__(**kwargs)
        self.query_linear=nn.Linear()
        self.key_linear=nn.Linear()
        self.value_linear=nn.Linear()



    def forward(self,x,edge_index:torch.Tensor):
        """

        """
import numpy as np
import torch
import torch.nn as nn

class TimeEncoder(nn.Module):
    def __init__(self,time_dim:int,parameter_requires_grad:bool=True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(TimeEncoder,self).__init__()
        self.time_dim=time_dim
        self.w=nn.Linear(1,time_dim)
        self.w.weight=nn.Parameter((torch.from_numpy(1/10**np.linspace(0,9,time_dim,dtype=np.float32))).reshape(time_dim,-1))
        self.w.bias=nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad=False
            self.w.bias.requires_grad=False

    def forward(self,timespan:torch.Tensor):
        """
        compute time encodings of time in timespan
        Input:
            timespan: [B,1] or [B,N,1]
        Output:
            updated_timespan: [B,N,time_dim] or [B,N,time_dim]
        """
        output=torch.cos(self.w(timespan)) # [B,N,time_dim] or [B,N,time_dim]
        return output


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
        x: [N,1]
        edge_index: [B,2,E]
        """
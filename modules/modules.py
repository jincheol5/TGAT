import numpy as np
import torch
import torch.nn as nn
from .graph_utils import TemporalGraphData

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
            updated_timespan: [B,time_dim] or [B,N,time_dim]
        """
        output=torch.cos(self.w(timespan)) # [B,time_dim] or [B,N,time_dim]
        return output


class TemporalAttentionLayer(nn.Module):
    def __init__(self,
            latent_dim:int,
            time_dim:int,
            n_head:int=1
        ):
        super().__init__()
        self.latent_dim=latent_dim
        self.time_dim=time_dim
        self.latent_dim=latent_dim
        self.qkv_dim=self.latent_dim+self.time_dim
        self.multi_head_attn=nn.MultiheadAttention(
            embed_dim=self.qkv_dim,
            kdim=self.qkv_dim,
            vdim=self.qkv_dim,
            num_heads=n_head
        )
        self.FFN=nn.Sequential(
            nn.Linear(
                in_features=self.qkv_dim+self.latent_dim,
                out_features=self.latent_dim
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=self.latent_dim,
                out_features=self.latent_dim
            )
        )

    def forward(self,
            tar_ft:torch.Tensor,
            tar_t_ft:torch.Tensor,
            n_ft:torch.Tensor,
            n_t_ft:torch.Tensor,
            n_mask:torch.Tensor
        ):
        """
        Input:
            tar_ft: [B,latent_dim]
            tar_t_ft: [B,time_dim]
            n_ft: [B,N,latent_dim]
            n_t_ft: [B,N,time_dim]
            n_mask: [B,N], True = valid neighbor
        Output:

        """
        ### set init
        tar_ft=tar_ft.unsqueeze(tar_ft,dim=1) # -> [B,1,latent_dim]
        tar_t_ft=tar_t_ft.unsqueeze(tar_t_ft,dim=1) # -> [B,1,time_dim]

        query=torch.cat(
            [tar_ft,tar_t_ft],
            dim=2
        ) # -> [B,1,latent_dim+time_dim]
        key=torch.cat(
            [n_ft,n_t_ft],
            dim=2
        ) # -> [B,N,latent_dim+time_dim]
        value=torch.cat(
            [n_ft,n_t_ft],
            dim=2
        ) # -> [B,N,latent_dim+time_dim]

        ### set to [L,B,D]
        query=query.permute([1,0,2]) # -> [1,B,latent_dim+time_dim] 
        key=key.permute([1,0,2]) # -> [N,B,latent_dim+time_dim] 
        value=value.permute([1,0,2]) # -> [N,B,latent_dim+time_dim] 

        ### transform n_mask for nn.MultiheadAttention's key_padding_mask
        # key_padding_mask에서는 True가 padding될 neighbor을 의미
        key_padding_mask=~n_mask

        ### Compute mask of which target nodes have no valid neighbors
        # tensor.all() -> 모든 값이 true인지 검사하는 함수
        # 이웃이 하나도 없는 target node 의 경우, attn 수행을 위해  첫 번째 이웃 노드를 임시로 유효하게 수정 (fake neighbor)   
        # fake neighbor 에만 attn이 집중되도록 강제
        # 이후 처리 
        invalid_neighborhood_mask=key_padding_mask.all(dim=1,keepdim=True) # [B,1], true=유효 neighbor 없음, false=유효 neighbor 존재
        key_padding_mask[invalid_neighborhood_mask.squeeze(),0]=False 

        ### Multi-head attention
        attn_output,_=self.multi_head_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask
        ) # attn_output: [1,B,latent_dim+time_dim], attn_weight: [B,1,N]
        attn_output=attn_output.squeeze() # -> [B,latent_dim+time_dim]

        ### 이웃노드가 없는 target node의 attn 결과 0으로 후처리
        attn_output=attn_output.mask_fill(invalid_neighborhood_mask,0) # mask_fill: mask=True인 위치를 value로 덮어쓰기

        ### FFN
        tar_ft=tar_ft.squeeze() # -> [B,latent_dim]
        ffn_input=torch.cat(
            [attn_output,tar_ft],
            dim=-1
        ) # -> [B,latent_dim+time_dim||latent_dim]
        output=self.FFN(ffn_input) # [B,latent_dim]
        return output

class GraphEmbedding(nn.Module):
    def __init__(self,
            latent_dim:int,
            time_dim:int,
            n_head:int=1,
            n_layer:int=1,
            data:TemporalGraphData=None
        ):
        super().__init__()
        self.data=data
        self.attn_layers=torch.nn.ModuleList([
            TemporalAttentionLayer(
                latent_dim=latent_dim,
                time_dim=time_dim,
                n_head=n_head
            )
        for _ in range(n_layer)])

    def compute_embedding(self,batch_tar,n_layer):
        """
        Input:
            batch_tar
            n_layer
        """
    
    def aggregate(self):
        """
        """
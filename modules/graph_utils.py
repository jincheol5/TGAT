import torch
import numpy as np

"""
To Do List:
- 0번 노드 dummy node 처리 -> 항상 이웃 노드 아무것도 없도록
"""
class TemporalGraphData:
    """
    node_ft: dict of each node feature
        key: node_id
        value: feature tensor
    neighbor: dict of each node's neighbor id list
        key: node_id
        value: list of neighbor node id
    neighbor_t: dict of each node's neighbor interact timestamp
        key: node_id
        value: list of neighbor interact timestamp
    """
    def __init__(self,node_dim:int=32):
        self.node_ft={}
        self.neighbor={}
        self.neighbor_t={}
        self.node_dim=node_dim
        
        # init dummy node feature
        self.node_ft[0]=torch.zeros(self.node_dim)

    def update_graph(self,event:tuple):
        """
        Input:
            event: tuple (src,tar,timestamp)
        """
        src,tar,timestamp=event
        if src not in self.node_ft:
            self.node_ft[src]=torch.ones(self.node_dim)
        if tar not in self.node_ft:
            self.node_ft[tar]=torch.ones(self.node_dim)
        if tar not in self.neighbor:
            self.neighbor[tar]=[]
            self.neighbor_t[tar]=[]
        self.neighbor[tar].append(src)
        self.neighbor_t[tar].append(timestamp)

    def find_temporal_neighbor(self,tar,cut_time):
        """
        """
        if tar not in self.neighbor or tar==0:
            return [],[]
        t_np=np.array(self.neighbor_t[tar])
        idx=np.searchsorted(t_np,cut_time,side="left")
        return self.neighbor[tar][:idx],self.neighbor_t[tar][:idx]

    def get_batch_data_for_embedding(self,batch_tar,batch_t):
        """
        Input:
            batch_tar: [B,]
            batch_t: [B,]
        Output
            batch_tar_ts: [B,]
            batch_n: [B,N]
            batch_n_t: [B,N]
            batch_n_ts: [B,N], 이웃 노드들과의 timespan
            batch_n_mask: [B,N]
            B = batch size
            N = max neighbor in batch
        """
        temporal_n=[
            self.find_temporal_neighbor(
                tar=tar.item(),
                cut_time=timestamp.item()
            )
            for tar,timestamp in zip(batch_tar,batch_t)
        ]

        n_list=[result[0] for result in temporal_n]
        n_t_list=[result[1] for result in temporal_n]
        batch_size=batch_tar.size(0)
        max_n=max(len(n) for n in n_list)

        batch_tar_ts=torch.zeros((batch_size,),dtype=torch.float32) # [B,]
        batch_n=torch.zeros((batch_size,max_n),dtype=torch.long) # [B,N]
        batch_n_t=torch.zeros((batch_size,max_n),dtype=torch.float32) # [B,N]
        batch_n_ts = torch.zeros((batch_size,max_n),dtype=torch.float32) # [B,N]
        batch_n_mask=torch.zeros((batch_size,max_n),dtype=torch.bool) # [B,N]

        for idx,(neighbors,timestamps) in enumerate(zip(n_list,n_t_list)):
            n_len=len(neighbors)
            if n_len==0:
                continue
            neighbors_tensor=torch.tensor(neighbors,dtype=torch.long)
            timestamps_tensor=torch.tensor(timestamps,dtype=torch.float32)

            batch_n[idx,:n_len]=neighbors_tensor
            batch_n_t[idx,:n_len]=timestamps_tensor

            batch_n_ts[idx,:n_len]=torch.abs(
                batch_n_t[idx]-timestamps_tensor
            )
            batch_n_mask[idx,:n_len]=True

        return {
            "batch_tar_ts": batch_tar_ts,
            "batch_n": batch_n,
            "batch_n_t": batch_n_t,
            "batch_n_ts": batch_n_ts,
            "batch_n_mask" : batch_n_mask
        }

    def get_batch_tar_feature(self,batch_tar):
        """
        Input:
            batch_tar: [B,]
        """
        batch_tar_ft=torch.stack(
            [self.node_ft[tar_id.item()] for tar_id in batch_tar],
            dim=0
        ) # [B,node_dim]
        return batch_tar_ft

    def get_batch_n_feature(self,batch_n,batch_n_mask):
        """
        Input:
            batch_n: [B,N]
            batch_n_mask: [B,N]
        """
        padding_ft=torch.zeros(self.node_dim)
        batch_n_ft=torch.stack([
            torch.stack([
                self.node_ft[node_id.item()]
                if is_valid.item()
                else padding_ft
                for node_id,is_valid in zip(neighbors,masks)
            ],dim=0)
            for neighbors,masks in zip(batch_n,batch_n_mask)
        ],dim=0) # [B,N,node_dim]
        return batch_n_ft

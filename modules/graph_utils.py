import torch
import numpy as np

"""
To Do List:
- batch 내의 같은 timestamp 이벤트들에 대해서 구조 정보 어떻게 처리할 지, np.searchsorted 함수 확인해보기 

"""
class TemporalGraphData:
    """
    node_feature: dict of each node feature
        key: node_id
        value: feature tensor
    neighbor: dict of each node's neighbor id list
        key: node_id
        value: list of neighbor node id
    ts: dict of each node's neighbor interact timestamp
        key: node_id
        value: list of neighbor interact timestamp
    """
    def __init__(self,latent_dim:int=32):
        self.node_feature={}
        self.neighbor={}
        self.ts={}
        self.latent_dim=latent_dim

    def update_graph(self,event:tuple):
        """
        Input:
            event: tuple (src,tar,timestamp)
        """
        src,tar,timestamp=event
        if src not in self.node_feature:
            self.node_feature[src]=torch.ones(self.latent_dim)
        if tar not in self.node_feature:
            self.node_feature[tar]=torch.ones(self.latent_dim)
        if tar not in self.neighbor:
            self.neighbor[tar]=[]
            self.ts[tar]=[]
        self.neighbor[tar].append(src)
        self.ts[tar].append(timestamp)

    def find_temporal_neighbor(self,tar,cut_time):
        """
        """
        if tar not in self.neighbor:
            return [],[]
        ts_np=np.array(self.ts[tar])
        idx=np.searchsorted(ts_np,cut_time,side="left")
        return self.neighbor[tar][:idx],self.ts[tar][:idx]

    def get_batch_data_for_embedding(self,batch_tar:list,batch_cut_time:list):
        """
        Input
            batch_tar: List of target node id
            batch_cut_time: List of event timestamp
        Output
            batch_n: [B,N]
            batch_ts: [B,N]
            batch_timespan: [B,N]
            batch_n_mask: [B,N]
            B = batch size
            N = max neighbor in batch
        """
        temporal_n=[
            self.find_temporal_neighbor(tar=tar,cut_time=cut_time)
            for tar,cut_time in zip(batch_tar,batch_cut_time)
        ]
        n_list=[result[0] for result in temporal_n]
        ts_list=[result[1] for result in temporal_n]

        batch_size=len(batch_tar)
        max_n=max(len(n) for n in n_list)

        batch_n=torch.zeros((batch_size,max_n),dtype=torch.long)
        batch_ts=torch.zeros((batch_size,max_n),dtype=torch.float32)
        batch_timespan = torch.zeros((batch_size,max_n),dtype=torch.float32)
        batch_n_mask=torch.zeros((batch_size,max_n),dtype=torch.bool)

        for idx,(neighbors,timestamps) in enumerate(zip(n_list,ts_list)):
            n_len=len(neighbors)
            if n_len==0:
                continue
            neighbors_tensor=torch.tensor(neighbors,dtype=torch.long)
            timestamps_tensor=torch.tensor(timestamps,dtype=torch.float32)

            batch_n[idx,:n_len]=neighbors_tensor
            batch_ts[idx,:n_len]=timestamps_tensor

            batch_timespan[idx,:n_len]=torch.abs(
                torch.tensor(batch_cut_time[idx],dtype=torch.float32)
                -timestamps_tensor
            )
            batch_n_mask[idx,:n_len]=True
        return batch_n,batch_ts,batch_timespan,batch_n_mask



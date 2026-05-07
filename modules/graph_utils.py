import torch

class TemporalGraphData:
    """
    node_feature: dict of each node feature
        key: node_id
        value: feature tensor
    adj: dict of each node's neighbor info, sorted by time asc 
        key: node_id
        value: list of tuple (src,time)
    """
    def __init__(self,latent_dim:int=32):
        self.node_feature={}
        self.adj={}
        self.latent_dim=latent_dim

    def update_graph(self,event:tuple):
        """
        Input:
            event: tuple (src,tar,time)
        """
        src,tar,time=event
        if src not in self.node_feature:
            self.node_feature[src]=torch.ones(self.latent_dim)
        if tar not in self.node_feature:
            self.node_feature[tar]=torch.ones(self.latent_dim)
        if tar not in self.adj:
            self.adj[tar]=[]
        self.adj[tar].append((src,time))
        self.adj[tar].sort(key=lambda x: x[1])

    def get_data_for_embedding(self,batch_events:list):
        """
        Input:
            batch_events: list of event tuple (src,tar,time)
        Output:
            batch_tar_ft: [B,latent_dim]
            batch_tar_ts: [B,1]
            batch_n_ft: [B,N,latent_dim]
            batch_n_ts: [B,N,1]
            batch_n_mask: [B,N,]
        """
        max_n=0
        tar_ft_list=[]
        n_list=[]
        ts_list=[]
        for event in batch_events:
            _,tar,time=event
            if tar not in self.node_feature:
                tar_ft=torch.ones(self.latent_dim)
            else:
                tar_ft=self.node_feature[tar]
            tar_ft_list.append(tar_ft)
            neighbors=[]
            timespans=[]
            for src,timestamp in self.adj.get(tar,[])[::-1]: # 역순회, tar 이웃노드들 없는 경우 [] 반환
                if src not in neighbors: # 같은 src의 경우 최신 시간값으로 계산
                    neighbors.append(src)
                    timespans.append(abs(time-timestamp))
            if max_n<len(neighbors):
                max_n=len(neighbors)
            n_list.append(neighbors)
            ts_list.append(timespans)
            self.update_graph(event=event) # 현재 event 이전 정보들만 참고할 수 있도록 함

        batch_n_ft_list=[]
        batch_n_ts_list=[]
        batch_n_mask_list=[]
        for neighbors,timespans in zip(n_list,ts_list):
            if len(neighbors)==0:  # 이웃이 없는 경우 빈 tensor 생성 (형태 정보만 유지)
                n_ft=torch.zeros((0,self.latent_dim),dtype=torch.float32)
                n_ts=torch.zeros((0,1),dtype=torch.float32)
            else:
                n_ft_list=[self.node_feature[n] for n in neighbors]
                n_ft=torch.stack(n_ft_list) # [N,latent_dim]
                n_ts=torch.tensor(timespans,dtype=torch.float32).unsqueeze(-1) # [N,1]

            # padding
            valid_row=n_ft.size(0)
            if valid_row<max_n:
                pad_rows=max_n-n_ft.size(0)
                n_ft_padding=torch.zeros(
                    (pad_rows,n_ft.size(1)),
                    dtype=n_ft.dtype
                )
                n_ft=torch.cat([n_ft,n_ft_padding],dim=0)
                n_ts_padding=torch.zeros(
                    (pad_rows,1),
                    dtype=n_ft.dtype
                )
                n_ts=torch.cat([n_ts,n_ts_padding],dim=0)

            # padding mask
            n_mask=torch.zeros(max_n,dtype=torch.bool)
            n_mask[:valid_row]=True # [N,]
            batch_n_ft_list.append(n_ft)
            batch_n_ts_list.append(n_ts)
            batch_n_mask_list.append(n_mask)
        batch_tar_ft=torch.stack(tar_ft_list) # [B,latent_dim]
        batch_tar_ts=torch.zeros((batch_tar_ft.size(0),1),dtype=torch.float32) # [B,1]
        batch_n_ft=torch.stack(batch_n_ft_list) # [B,N,latent_dim]
        batch_n_ts=torch.stack(batch_n_ts_list) # [B,N,1]
        batch_n_mask=torch.stack(batch_n_mask_list) # [B,N,]
        return batch_tar_ft,batch_tar_ts,batch_n_ft,batch_n_ts,batch_n_mask
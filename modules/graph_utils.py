import torch

class TemporalGraphData:
    """
    node_feature: [N,latent_dim]
    edge_index: [2,E]
    edge_time: [E,]
    """
    def __init__(self):
        pass

    def update_batch_events(self,batch_events:list):
        """
        batch_events: list of edge_event tuple (src,tar,time)
        """
        # 새로운 batch의 edge 정보 추출
        src=[e[0] for e in batch_events]
        tar=[e[1] for e in batch_events]
        time=[e[2] for e in batch_events]
        batch_edge_index=torch.tensor([src,tar],dtype=torch.long)
        batch_edge_time=torch.tensor(time,dtype=torch.long)
        
        # 기존 edge_index 존재 여부 확인
        if hasattr(self,'edge_index') and self.edge_index is not None:
            # 기존 edge_index와 새로운 batch 합치기
            self.edge_index=torch.cat([self.edge_index,batch_edge_index],dim=1)
            self.edge_time=torch.cat([self.edge_time,batch_edge_time],dim=0)
        else:
            # 새로 생성
            self.edge_index=batch_edge_index
            self.edge_time=batch_edge_time
        
        # 중복 제거 (같은 src,tar의 경우 최신 time만 유지)
        unique_edges={}
        for event_idx in range(self.edge_index.size(1)):
            key=(self.edge_index[0,event_idx].item(),self.edge_index[1,event_idx].item()) # key=(src,tar)
            if key not in unique_edges or self.edge_time[event_idx]>unique_edges[key][1]:
                unique_edges[key]=(event_idx,self.edge_time[event_idx])
        unique_indices=torch.tensor([idx for idx,_ in unique_edges.values()],dtype=torch.long)
        self.edge_index=self.edge_index[:,unique_indices]
        self.edge_time=self.edge_time[unique_indices]
        
        # 첫 번째 행 기준으로 오름차순 정렬
        sort_idx=torch.argsort(self.edge_index[0])
        self.edge_index=self.edge_index[:,sort_idx]
        self.edge_time=self.edge_time[sort_idx]
    
    def get_edge_index(self):
        return self.edge_index

    def get_edge_time(self):
        return self.edge_time


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

    def update_graph(self,batch_events:list):
        """
        batch_events: list of event tuple (src,tar,time)
        """
        for event in batch_events:
            src,tar,time=event
            if src not in self.node_feature:
                self.node_feature[src]=torch.ones(self.latent_dim)
            if tar not in self.node_feature:
                self.node_feature[tar]=torch.ones(self.latent_dim)
            if tar not in self.adj:
                self.adj[tar]=[]
            self.adj[tar].append((src,time))
            self.adj[tar].sort(key=lambda x: x[1])

    def get_neighbor_ft(self,batch_events:list):
        """
        batch_events: list of event tuple (src,tar,time)

        Returns:
            neighbor_ft: tensor of shape [B, N, latent_dim]
            mask: tensor of shape [B, N], where 1 indicates a real neighbor and 0 indicates padding

        N is the maximum number of neighbors among all tar nodes in the batch.
        """
        batch_size = len(batch_events)
        neighbor_lists = []
        max_neighbors = 0

        for _, tar, _ in batch_events:
            seen = set()
            unique_neighbors = []
            for src, _ in self.adj.get(tar, []):
                if src not in seen:
                    seen.add(src)
                    unique_neighbors.append(src)
            neighbor_lists.append(unique_neighbors)
            if len(unique_neighbors) > max_neighbors:
                max_neighbors = len(unique_neighbors)

        neighbor_ft = torch.zeros((batch_size, max_neighbors, self.latent_dim))
        mask = torch.zeros((batch_size, max_neighbors), dtype=torch.bool)

        for i, neighbors in enumerate(neighbor_lists):
            for j, src in enumerate(neighbors):
                feature = self.node_feature.get(src, torch.zeros(self.latent_dim))
                neighbor_ft[i, j] = feature
                mask[i, j] = True

        return neighbor_ft, mask

    def get_timespan_for_embedding(self,batch_events:list):
        """
        Input:
            batch_events: list of event tuple (src,tar,time)
        Output:
            tar_timespan: [B,1]
            neighbor_timespan: [B,N,1]
        """

    def get_data_for_embedding(self,batch_events:list):
        """
        Input:
            batch_events: list of event tuple (src,tar,time)
        Output:
            tar_ft: [B,latent]
            neighbor_ft: [B,N,latent_dim]
        """

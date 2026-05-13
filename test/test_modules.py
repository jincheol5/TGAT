import argparse
import torch
from modules import TemporalGraphData
from modules import TemporalGraphEmbedding

def test_modules(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. TemporalGraphEmbedding
            """
            eventstream=[
                (3,7,1),
                (3,6,2),
                (1,4,3),
                (1,3,4),

                (1,2,5),
                (2,7,5),
                (1,4,7),
                (4,7,7),

                (4,5,8),
                (1,3,9)
            ]
            data=TemporalGraphData(node_dim=4)
            data.update_graph(batch_events=eventstream)
            batch_tar=torch.tensor([2,3,7],dtype=torch.long)
            batch_t=torch.tensor([10.0,10.0,10.0],dtype=torch.float32)
            n_layer=2
            model=TemporalGraphEmbedding(
                node_dim=4,
                latent_dim=4,
                time_dim=4,
                n_head=4,
                n_layer=n_layer,
                data=data
            )
            batch_tar_ft=model.compute_embedding(
                batch_tar=batch_tar,
                batch_t=batch_t,
                n_layer=n_layer
            )
            print(f"embedded batch_tar_ft:")
            print(f"{batch_tar_ft}")

if __name__=="__main__":
    """
    Execute test_modules
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_num",type=int,default=1)
    args=parser.parse_args()
    test_config={
        'test_num':args.test_num
    }
    test_modules(**test_config)
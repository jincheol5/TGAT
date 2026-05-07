import argparse
from modules import TemporalGraphData

def test_graph_utils(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. TemporalGraphData.update_graph()
            """

        case 2:
            """
            Test. TemporalGraphData.get_data_for_embedding()
            """
            graph=TemporalGraphData(latent_dim=4)
            batch_events=[
                (1,2,10), 
                (2,3,20), 
                (1,3,25)
            ]
            batch_tar_ft,batch_tar_ts,batch_n_ft,batch_n_ts,batch_n_mask=graph.get_data_for_embedding(batch_events)
            
            for idx in range(len(batch_events)):
                print(f"{idx+1} batch tar_ft:")
                print(f"{batch_tar_ft[idx]}",end="\n\n")

                print(f"{idx+1} batch tar_ts:")
                print(f"{batch_tar_ts[idx]}",end="\n\n")

                print(f"{idx+1} batch n_ft:")
                print(f"{batch_n_ft[idx]}",end="\n\n")

                print(f"{idx+1} batch n_ts:")
                print(f"{batch_n_ts[idx]}",end="\n\n")

                print(f"{idx+1} batch n_mask:")
                print(f"{batch_n_mask[idx]}",end="\n\n")

if __name__=="__main__":
    """
    Execute test_graph_utils
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_num",type=int,default=1)
    args=parser.parse_args()
    test_config={
        'test_num':args.test_num
    }
    test_graph_utils(**test_config)
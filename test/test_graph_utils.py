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
            Test. TemporalGraphData.update_edge_index()
            """

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
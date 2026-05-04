import argparse
from modules import TemporalGraphData

def test_graph_utils(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. TemporalGraphData.update_edge_index()
            """
            # 초기 event stream 생성
            initial_events=[
                (0,1,100),
                (1,2,110),
                (0,2,120),
                (1,3,130),
            ]
            
            # TemporalGraphData 초기화
            graph=TemporalGraphData()
            graph.update_batch_events(batch_events=initial_events)
            print("=== Initial Graph Data ===")
            print(f"edge_index shape: {graph.edge_index.shape}")
            print(f"edge_index:\n{graph.edge_index}")
            print(f"edge_time shape: {graph.edge_time.shape}")
            print(f"edge_time: {graph.edge_time}")
            
            # 새로운 batch 추가
            batch_events=[
                (0,1,150),  # 기존 edge 중복 (최신 time으로 업데이트)
                (2,3,140),  # 새로운 edge
                (1,2,160),  # 기존 edge 중복 (최신 time으로 업데이트)
            ]
            
            print("\n=== Adding Batch Events ===")
            print(f"batch_events: {batch_events}")
            graph.update_batch_events(batch_events=batch_events)
            
            print("\n=== Updated Graph Data ===")
            print(f"edge_index shape: {graph.edge_index.shape}")
            print(f"edge_index:\n{graph.edge_index}")
            print(f"edge_time shape: {graph.edge_time.shape}")
            print(f"edge_time: {graph.edge_time}")
            
            # 검증
            print("\n=== Verification ===")
            print(f"No duplicates: {graph.edge_index.shape[1] == len(set([(graph.edge_index[0,i].item(), graph.edge_index[1,i].item()) for i in range(graph.edge_index.shape[1])]))} ")
            print(f"Sorted by src: {all(graph.edge_index[0,i] <= graph.edge_index[0,i+1] for i in range(graph.edge_index.shape[1]-1))}")

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
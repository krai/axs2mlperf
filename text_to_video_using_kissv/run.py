import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed MLPerf Runner")

    # Integer arguments for node configuration
    parser.add_argument(
        "--nodes", 
        type=int, 
        required=True, 
        help="Total number of nodes in the cluster"
    )
    parser.add_argument(
        "--node_idx", 
        type=int, 
        required=True, 
        help="The index of the current node (0-indexed)"
    )

    # String arguments for the specific commands
    parser.add_argument(
        "--docker_cmd", 
        type=str, 
        required=True, 
        help="The full Docker run command to execute"
    )
    parser.add_argument(
        "--mlperf_cmd", 
        type=str, 
        required=True, 
        help="The specific MLPerf command to run"
    )

    return parser.parse_args()

if __name__ == "__main__":
    print(parse_args())
    breakpoint()
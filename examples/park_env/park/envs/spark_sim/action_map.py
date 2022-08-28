
class two_way_unordered_map(object):
    def __init__(self):
        self.map = {}
        self.inverse_map = {}

    def __setitem__(self, key, value):
        self.map[key] = value
        self.inverse_map[value] = key
        # keys and values should be unique
        assert len(self.map) == len(self.inverse_map)

    def __getitem__(self, key):
        return self.map[key]

    def __len__(self):
        return len(self.map)


def compute_act_map(job_dags):
    # translate action ~ [0, num_nodes_in_all_dags) to node object
    action_map = two_way_unordered_map()
    action = 0
    for job_dag in job_dags:
        for node in job_dag.nodes:
            action_map[action] = node
            action += 1
    return action_map

def get_frontier_acts(job_dags):
    # O(num_total_nodes)
    frontier_actions = []
    base = 0
    for job_dag in job_dags:
        for node_idx in job_dag.frontier_nodes:
            frontier_actions.append(base + node_idx)
        base += job_dag.num_nodes
    return frontier_actions
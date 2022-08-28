"""
Invoke these two functions when adding/removing jobs
in the system. Every time in observation, we only need
to update the features (e.g., don't have to keep changing
the adjacency matrix).
"""

def add_job_in_graph(graph, job_dag):
    for node in job_dag.nodes:
        graph.update_nodes({node: None})
    for node in job_dag.nodes:
        for child in node.child_nodes:
            graph.update_edges({(node, child): None})
    # Note: node features will be added during observation


def remove_job_from_graph(graph, job_dag):
    for node in job_dag.nodes:
        graph.remove_nodes([node])
        # Note: edges will be removed
        # automatically by networkx

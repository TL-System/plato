from collections import OrderedDict


class ExecutorCommit(object):
    def __init__(self):
        # {node/job_dag -> ordered{node -> amount}}
        self.commit = {}
        # {node -> amount}
        self.node_commit = {}
        # {node -> set(nodes/job_dags)}
        self.backward_map = {}

    def __getitem__(self, source):
        return self.commit[source]

    def add(self, source, node, amount):
        # source can be node or job
        # node: executors continuously free up
        # job: free executors

        # add foward connection
        if node not in self.commit[source]:
            self.commit[source][node] = 0
        # add node commit
        self.commit[source][node] += amount
        # add to record of total commit on node
        self.node_commit[node] += amount
        # add backward connection
        self.backward_map[node].add(source)

    def pop(self, source):
        # implicitly assert source in self.commit
        # implicitly assert len(self.commit[source]) > 0

        # find the node in the map
        node = next(iter(self.commit[source]))
        
        # deduct one commitment
        self.commit[source][node] -= 1
        self.node_commit[node] -= 1
        assert self.commit[source][node] >= 0
        assert self.node_commit[node] >= 0

        # remove commitment on job if exhausted
        if self.commit[source][node] == 0:
            del self.commit[source][node]
            self.backward_map[node].remove(source)

        return node

    def add_job(self, job_dag):
        # add commit entry to the map
        self.commit[job_dag] = OrderedDict()
        for node in job_dag.nodes:
            self.commit[node] = OrderedDict()
            self.node_commit[node] = 0
            self.backward_map[node] = set()

    def remove_job(self, job_dag):
        # when removing jobs, the commiment should be all satisfied
        assert len(self.commit[job_dag]) == 0
        del self.commit[job_dag]

        # clean up commitment to the job
        for node in job_dag.nodes:
            # the executors should all move out
            assert len(self.commit[node]) == 0
            del self.commit[node]

            for source in self.backward_map[node]:
                # remove forward link
                del self.commit[source][node]
            # remove backward link
            del self.backward_map[node]
            # remove node commit records
            del self.node_commit[node]

    def reset(self):
        self.commit = {}
        self.node_commit = {}
        self.backward_map = {}
        # for agent to make void action
        self.commit[None] = OrderedDict()
        self.node_commit[None] = 0
        self.backward_map[None] = set()

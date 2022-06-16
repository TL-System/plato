
class MovingExecutors(object):
    def __init__(self):
        # executor -> node
        self.moving_executors = {}
        # node -> set(executors)
        self.node_track = {}
    
    def __contains__(self, executor):
        return executor in self.moving_executors

    def __getitem__(self, executor):
        return self.moving_executors[executor]

    def __len__(self):
        return len(self.moving_executors)

    def add(self, executor, node):
        # detach the executor from old job
        executor.detach_job()
        # keep track of moving executor
        self.moving_executors[executor] = node
        self.node_track[node].add(executor)

    def pop(self, executor):
        if executor in self.moving_executors:
            node = self.moving_executors[executor]
            self.node_track[node].remove(executor)
            del self.moving_executors[executor]
        else:
            # job is completed by the time
            # executor arrives
            node = None
        return node

    def count(self, node):
        return len(self.node_track[node])

    def add_job(self, job_dag):
        for node in job_dag.nodes:
            self.node_track[node] = set()

    def remove_job(self, job_dag):
        for node in job_dag.nodes:
            for executor in self.node_track[node]:
                del self.moving_executors[executor]
            del self.node_track[node]

    def reset(self):
       self.moving_executors = {}
       self.node_track = {}

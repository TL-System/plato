class Executor(object):

    def __init__(self, idx):
        self.idx = idx
        self.task = None
        self.node = None
        self.job_dag = None

    def detach_node(self):
        if self.node is not None and \
           self in self.node.executors:
            self.node.executors.remove(self)
        self.node = None
        self.task = None

    def detach_job(self):
        if self.job_dag is not None and \
           self in self.job_dag.executors:
            self.job_dag.executors.remove(self)
        self.job_dag = None
        self.detach_node()

    def reset(self):
        self.task = None
        self.node = None
        self.job_dag = None

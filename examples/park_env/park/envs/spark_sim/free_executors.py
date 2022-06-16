from park.utils.ordered_set import OrderedSet


class FreeExecutors(object):
    def __init__(self, executors):
        self.free_executors = {}
        self.free_executors[None] = OrderedSet()
        for executor in executors:
            self.free_executors[None].add(executor)

    def __getitem__(self, job):
        return self.free_executors[job]

    def contain_executor(self, job, executor):
        if executor in self.free_executors[job]:
            return True
        else:
            return False

    def pop(self, job):
        executor = next(iter(self.free_executors[job]))
        self.free_executors[job].remove(executor)
        return executor

    def add(self, job, executor):
        if job is None:
            executor.detach_job()
        else:
            executor.detach_node()
        self.free_executors[job].add(executor)

    def remove(self, executor):
        self.free_executors[executor.job_dag].remove(executor)

    def add_job(self, job):
        self.free_executors[job] = OrderedSet()

    def remove_job(self, job):
        # put all free executors to global free pool
        for executor in self.free_executors[job]:
            executor.detach_job()
            self.free_executors[None].add(executor)
        del self.free_executors[job]

    def reset(self, executors):
        self.free_executors = {}
        self.free_executors[None] = OrderedSet()
        for executor in executors:
            self.free_executors[None].add(executor)

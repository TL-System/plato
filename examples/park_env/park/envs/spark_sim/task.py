import numpy as np

class Task(object):
    def __init__(self, idx, rough_duration, wall_time):
        self.idx = idx
        self.wall_time = wall_time

        self.duration = rough_duration

        # uninitialized
        self.start_time = np.nan
        self.finish_time = np.nan
        self.executor = None
        self.node = None

    def schedule(self, start_time, duration, executor):
        assert np.isnan(self.start_time)
        assert np.isnan(self.finish_time)
        assert self.executor is None

        self.start_time = start_time
        self.duration = duration
        self.finish_time = self.start_time + duration

        # bind the executor to the task and
        # the task with the given executor
        self.executor = executor
        self.executor.task = self
        self.executor.node = self.node
        self.executor.job_dag = self.node.job_dag

    def get_duration(self):
        # get task duration lazily
        if np.isnan(self.start_time):
            # task not scheduled yet
            return self.duration
        elif self.wall_time.curr_time < self.start_time:
            # task not started yet
            return self.duration
        else:
            # task running or completed
            duration = max(0,
                self.finish_time - self.wall_time.curr_time)
            return duration

    def reset(self):
        self.start_time = np.nan
        self.finish_time = np.nan
        self.executor = None

from collections import deque


class Server(object):
    def __init__(self, server_id, service_rate, wall_time):
        self.server_id = server_id
        self.service_rate = service_rate
        self.wall_time = wall_time
        self.queue = deque()
        self.curr_job = None

    def schedule(self, job):
        self.queue.append(job)
        job.server = self

    def process(self):
        # if the server is currently idle (no current
        # job or current job is done), and there are jobs
        # in the queue, then FIFO process a job
        if (self.curr_job is None or \
           self.curr_job.finish_time <= self.wall_time.curr_time) \
           and len(self.queue) > 0:

            self.curr_job = self.queue.popleft()
            duration = int(self.curr_job.size / self.service_rate)
            self.curr_job.start_time = self.wall_time.curr_time
            self.curr_job.finish_time = self.wall_time.curr_time + duration

            return self.curr_job

        else:
            return None

    def reset(self):
        self.queue = deque()
        self.curr_job = None

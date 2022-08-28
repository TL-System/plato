import numpy as np


class RewardCalculator(object):
    def __init__(self):
        self.job_dags = set()
        self.prev_time = 0

    def get_reward(self, job_dags, curr_time):
        reward = 0

        # add new job into the store of jobs
        for job_dag in job_dags:
            self.job_dags.add(job_dag)

        # now for all jobs (may have completed)
        # compute the elapsed time
        for job_dag in list(self.job_dags):
            reward -= (min(
                job_dag.completion_time,
                curr_time) - max(
                job_dag.start_time,
                self.prev_time))

            # if the job is done, remove it from the list
            if job_dag.completed:
                self.job_dags.remove(job_dag)

        self.prev_time = curr_time

        return reward

    def reset(self):
        self.job_dags.clear()
        self.prev_time = 0

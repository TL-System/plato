import numpy as np

from park import core, spaces, logger
from park.param import config
from park.utils import seeding
from park.envs.load_balance.job import Job
from park.envs.load_balance.job_generator import generate_job, generate_jobs
from park.envs.load_balance.server import Server
from park.envs.load_balance.timeline import Timeline
from park.envs.load_balance.wall_time import WallTime


class LoadBalanceEnv(core.Env):
    """
    Balance the load among n (default 10) heterogeneous servers
    to minimize  average job processing time (queuing delay).
    Jobs arrive according to a Poisson process and the job size
    distributes according to a Pareto distribution.

    * STATE *
        Current Load (total work waiting in the queue +
        remaining work currently being processed (if any) among n
        servers) and the incoming job size.
        The state is represented as a vector:
        [load_server_1, load_server_2, ..., load_server_n, job_size]

    * ACTIONS *
        Which server to assign the incoming job, represented as an
        integer in [0, n-1].

    * REWARD *
        Negative time elapsed for each job in the system since last action.
        For example, the virtual time was 0 for the last action, 4 jobs
        was in the system (either in the queue waiting or being processed),
        job 1 finished at time 1, job 2 finished at time 2.4 and job 3 and 4
        are still running at the next action. The next action is taken at
        time 5. Then the reward is - (1 * 1 + 1 * 2.4 + 2 * 5).
        Thus, the sum of the rewards would be negative of total
        (waiting + processing) time for all jobs.
    
    * REFERENCE *
        Figure 1a, Section 6.2 and Appendix J
        Variance Reduction for Reinforcement Learning in Input-Driven Environments. 
        H Mao, SB Venkatakrishnan, M Schwarzkopf, M Alizadeh.
        https://openreview.net/forum?id=Hyg1G2AqtQ
    
        Certain optimality properties of the first-come first-served discipline for g/g/s queues.
        DJ Daley.
        Stochastic Processes and their Applications, 25:301–308, 1987.
    """
    def __init__(self):
        # observation and action space
        self.setup_space()
        # random seed
        self.seed(config.seed)
        # global timer
        self.wall_time = WallTime()
        # uses priority queue
        self.timeline = Timeline()
        # total number of streaming jobs (can be very large)
        self.num_stream_jobs = config.num_stream_jobs
        # servers
        self.servers = self.initialize_servers(config.service_rates)
        # current incoming job to schedule
        self.incoming_job = None
        # finished jobs (for logging at the end)
        self.finished_jobs = []
        # reset environment (generate new jobs)
        self.reset()

    def generate_job(self):
        if self.num_stream_jobs_left > 0:
            dt, size = generate_job(self.np_random)
            t = self.wall_time.curr_time
            self.timeline.push(t + dt, size)
            self.num_stream_jobs_left -= 1

    def generate_jobs(self):
        all_t, all_size = generate_jobs(self.num_stream_jobs, self.np_random)
        for t, size in zip(all_t, all_size):
            self.timeline.push(t, size)

    def initialize(self):
        assert self.wall_time.curr_time == 0
        # generate a job
        self.generate_job()
        new_time, obj = self.timeline.pop()
        self.wall_time.update(new_time)
        assert isinstance(obj, int)  # a job arrival event
        size = obj
        self.incoming_job = Job(size, self.wall_time.curr_time)

    def initialize_servers(self, service_rates):
        servers = []
        for server_id in range(config.num_servers):
            server = Server(server_id, service_rates[server_id], self.wall_time)
            servers.append(server)
        return servers

    def observe(self):
        obs_arr = []
        # load on each server
        for server in self.servers:
            # queuing work
            load = sum(j.size for j in server.queue)
            if server.curr_job is not None:
                # remaining work currently being processed
                load += server.curr_job.finish_time - self.wall_time.curr_time
            # if the load is larger than observation threshold
            # report a warning
            if load > self.obs_high[server.server_id]:
                logger.warn('Server ' + str(server.server_id) + ' at time ' +
                             str(self.wall_time.curr_time) + ' has load ' + str(load) +
                             ' larger than obs_high ' + str(self.obs_high[server.server_id]))
                load = self.obs_high[server.server_id]
            obs_arr.append(load)

        # incoming job size
        if self.incoming_job is None:
            obs_arr.append(0)
        else:
            if self.incoming_job.size > self.obs_high[-1]:
                logger.warn('Incoming job at time ' + str(self.wall_time.curr_time) +
                              ' has size ' + str(self.incoming_job.size) +
                              ' larger than obs_high ' + str(self.obs_high[-1]))
                obs_arr.append(self.obs_high[-1])
            else:
                obs_arr.append(self.incoming_job.size)

        obs_arr = np.array(obs_arr)
        assert self.observation_space.contains(obs_arr)

        return obs_arr

    def reset(self):
        for server in self.servers:
            server.reset()
        self.wall_time.reset()
        self.timeline.reset()
        self.num_stream_jobs_left = self.num_stream_jobs
        assert self.num_stream_jobs_left > 0
        self.incoming_job = None
        self.finished_jobs = []
        # initialize environment (jump to first job arrival event)
        self.initialize()
        return self.observe()

    def seed(self, seed):
        self.np_random = seeding.np_random(seed)

    def setup_space(self):
        # Set up the observation and action space
        # The boundary of the space may change if the dynamics is changed
        # a warning message will show up every time e.g., the observation falls
        # out of the observation space
        self.obs_low = np.array([0] * (config.num_servers + 1))
        self.obs_high = np.array([config.load_balance_obs_high] * (config.num_servers + 1))
        self.observation_space = spaces.Box(
            low=self.obs_low, high=self.obs_high, dtype=np.float32)
        self.action_space = spaces.Discrete(config.num_servers)

    def step(self, action):

        # 0 <= action < num_servers
        assert self.action_space.contains(action)

        # schedule job to server
        self.servers[action].schedule(self.incoming_job)
        running_job = self.servers[action].process()
        if running_job is not None:
            self.timeline.push(running_job.finish_time, running_job)

        # erase incoming job
        self.incoming_job = None

        # generate next job
        self.generate_job()

        # set to compute reward from this time point
        reward = 0

        while len(self.timeline) > 0:

            new_time, obj = self.timeline.pop()

            # update reward
            num_active_jobs = sum(len(w.queue) for w in self.servers)
            for server in self.servers:
                if server.curr_job is not None:
                    assert server.curr_job.finish_time >= \
                           self.wall_time.curr_time  # curr job should be valid
                    num_active_jobs += 1
            reward -= (new_time - self.wall_time.curr_time) * num_active_jobs

            # tick time
            self.wall_time.update(new_time)

            if isinstance(obj, int):  # new job arrives
                size = obj
                self.incoming_job = Job(size, self.wall_time.curr_time)
                # break to consult agent
                break

            elif isinstance(obj, Job):  # job completion on server
                job = obj
                if not np.isinf(self.num_stream_jobs_left):
                    self.finished_jobs.append(job)
                else:
                    # don't store infinite streaming
                    # TODO: stream the complete job to some file
                    if len(self.finished_jobs) > 0:
                        self.finished_jobs[-1] +=1
                    else:
                        self.finished_jobs = [1]
                if job.server.curr_job == job:
                    # server's current job is done
                    job.server.curr_job = None
                running_job = job.server.process()
                if running_job is not None:
                    self.timeline.push(running_job.finish_time, running_job)

            else:
                print("illegal event type")
                exit(1)

        done = ((len(self.timeline) == 0) and \
               self.incoming_job is None)

        return self.observe(), reward, done, {'curr_time': self.wall_time.curr_time}

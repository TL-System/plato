import numpy as np
import copy
from collections import OrderedDict

from park import core, spaces, logger
from park.param import config
from park.utils import seeding
from park.utils.ordered_set import OrderedSet
from park.utils.directed_graph import DirectedGraph
from park.envs.spark_sim.action_map import compute_act_map, get_frontier_acts
from park.envs.spark_sim.reward_calculator import RewardCalculator
from park.envs.spark_sim.moving_executors import MovingExecutors
from park.envs.spark_sim.executor_commit import ExecutorCommit
from park.envs.spark_sim.free_executors import FreeExecutors
from park.envs.spark_sim.job_generator import generate_jobs
from park.envs.spark_sim.wall_time import WallTime
from park.envs.spark_sim.timeline import Timeline
from park.envs.spark_sim.executor import Executor
from park.envs.spark_sim.job_dag import JobDAG
from park.envs.spark_sim.node import Node
from park.envs.spark_sim.task import Task
from park.envs.spark_sim.job_graph import add_job_in_graph, remove_job_from_graph


class SparkSimEnv(core.Env):
    """
    A trace-driven simulator for the dynamics of the scheduling module in Apache Spark.
    The intricacies to closely simulate the system in reality are mainly (1) the "moving
    cost" of executors acorss jobs (due to the overhead of starting a new JVM); (2) the
    wave effect of running tasks of the same stage on an executor (overhead of loading
    data in the first wave of tasks); (3) the dimenishing speedup in job runtime when
    assigning more executors to a job. See reference for more details.

    * STATE *
        Graph type of observation. It consists of features associated with each node (
        a tensor of dimension n * m, where n is number of nodes, m is number of features),
        and adjacency matrix (a sparse 0-1 matrix of dimension n * n).
        The features on each node is
        [number_of_executors_currently_in_this_job, is_current_executor_local_to_this_job,
         number_of_free_executors, total_work_remaining_on_this_node,
         number_of_tasks_remaining_on_this_node]

    * ACTIONS *
        Two dimensional action, [node_idx_to_schedule_next, number_of_executors_to_assign]
        Note: the set of available nodes has to contain node_idx, and the number of
        executors to assign must not exceed the limit. Both the available set and the limit
        are provided in the (auxiliary) state.

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
        Section 6.2
        Learning Scheduling Algorithms for Data Processing Clusters
        H Mao, M Schwarzkopf, SB Venkatakrishnan, M Alizadeh
        https://arxiv.org/pdf/1810.01963.pdf
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

        # executors
        self.executors = OrderedSet()
        for exec_id in range(config.exec_cap):
            self.executors.add(Executor(exec_id))

        # free executors
        self.free_executors = FreeExecutors(self.executors)

        # moving executors
        self.moving_executors = MovingExecutors()

        # executor commit
        self.exec_commit = ExecutorCommit()

        # prevent agent keeps selecting the same node
        self.node_selected = set()

        # for computing reward at each step
        self.reward_calculator = RewardCalculator()

    def add_job(self, job_dag):
        self.moving_executors.add_job(job_dag)
        self.free_executors.add_job(job_dag)
        self.exec_commit.add_job(job_dag)
        add_job_in_graph(self.graph, job_dag)

    def assign_executor(self, executor, frontier_changed):
        if executor.node is not None and not executor.node.no_more_tasks:
            # keep working on the previous node
            task = executor.node.schedule(executor)
            self.timeline.push(task.finish_time, task)
        else:
            # need to move on to other nodes
            if frontier_changed:
                # frontier changed, need to consult all free executors
                # note: executor.job_dag might change after self.schedule()
                source_job = executor.job_dag
                if len(self.exec_commit[executor.node]) > 0:
                    # directly fulfill the commitment
                    self.exec_to_schedule = {executor}
                    self.schedule()
                else:
                    # free up the executor
                    self.free_executors.add(source_job, executor)
                # then consult all free executors
                self.exec_to_schedule = OrderedSet(self.free_executors[source_job])
                self.source_job = source_job
                self.num_source_exec = len(self.free_executors[source_job])
            else:
                # just need to schedule one current executor
                self.exec_to_schedule = {executor}
                # only care about executors on the node
                if len(self.exec_commit[executor.node]) > 0:
                    # directly fulfill the commitment
                    self.schedule()
                else:
                    # need to consult for ALL executors on the node
                    # Note: self.exec_to_schedule is immediate
                    #       self.num_source_exec is for commit
                    #       so len(self.exec_to_schedule) !=
                    #       self.num_source_exec can happen
                    self.source_job = executor.job_dag
                    self.num_source_exec = len(executor.node.executors)

    def backup_schedule(self, executor):
        backup_scheduled = False
        if executor.job_dag is not None:
            # first try to schedule on current job
            for node in executor.job_dag.frontier_nodes:
                if not self.saturated(node):
                    # greedily schedule a frontier node
                    task = node.schedule(executor)
                    self.timeline.push(task.finish_time, task)
                    backup_scheduled = True
                    break
        # then try to schedule on any available node
        if not backup_scheduled:
            schedulable_nodes = self.get_frontier_nodes()
            if len(schedulable_nodes) > 0:
                node = next(iter(schedulable_nodes))
                self.timeline.push(
                    self.wall_time.curr_time + config.moving_delay, executor)
                # keep track of moving executors
                self.moving_executors.add(executor, node)
                backup_scheduled = True
        # at this point if nothing available, leave executor idle
        if not backup_scheduled:
            self.free_executors.add(executor.job_dag, executor)

    def get_frontier_nodes(self):
        # frontier nodes := unsaturated nodes with all parent nodes saturated
        frontier_nodes = OrderedSet()
        for job_dag in self.job_dags:
            for node in job_dag.nodes:
                if not node in self.node_selected and not self.saturated(node):
                    parents_saturated = True
                    for parent_node in node.parent_nodes:
                        if not self.saturated(parent_node):
                            parents_saturated = False
                            break
                    if parents_saturated:
                        frontier_nodes.add(node)

        return frontier_nodes

    def get_executor_limits(self):
        # "minimum executor limit" for each job
        # executor limit := {job_dag -> int}
        executor_limit = {}

        for job_dag in self.job_dags:

            if self.source_job == job_dag:
                curr_exec = self.num_source_exec
            else:
                curr_exec = 0

            # note: this does not count in the commit and moving executors
            executor_limit[job_dag] = len(job_dag.executors) - curr_exec

        return executor_limit

    def observe(self):
        # valid set of nodes
        frontier_nodes = self.get_frontier_nodes()

        # sort out the exec_map (where the executors are)
        exec_map = {}
        for job_dag in self.job_dags:
            exec_map[job_dag] = len(job_dag.executors)
        # count in moving executors
        for node in self.moving_executors.moving_executors.values():
            exec_map[node.job_dag] += 1
        # count in executor commit
        for s in self.exec_commit.commit:
            if isinstance(s, JobDAG):
                j = s
            elif isinstance(s, Node):
                j = s.job_dag
            elif s is None:
                j = None
            else:
                print('source', s, 'unknown')
                exit(1)
            for n in self.exec_commit.commit[s]:
                if n is not None and n.job_dag != j:
                    exec_map[n.job_dag] += self.exec_commit.commit[s][n]

        for job_dag in self.job_dags:
            for node in job_dag.nodes:
                feature = np.zeros([6])
                # number of executors already in the job
                feature[0] = exec_map[job_dag]
                # source executor is from the current job (locality)
                feature[1] = job_dag is self.source_job
                # number of source executors
                feature[2] = self.num_source_exec
                # remaining number of tasks in the node
                feature[3] = node.num_tasks - node.next_task_idx
                # average task duration of the node
                feature[4] = node.tasks[-1].duration
                # is the current node valid
                feature[5] = node in frontier_nodes

                # update feature in observation
                self.graph.update_nodes({node: feature})

        # update mask in the action space
        self.action_space[0].update_valid_set(frontier_nodes)

        # return the graph as observation
        obs = self.graph
        assert self.observation_space.contains(obs)

        return obs

    def saturated(self, node):
        # frontier nodes := unsaturated nodes with all parent nodes saturated
        anticipated_task_idx = node.next_task_idx + \
           self.exec_commit.node_commit[node] + \
           self.moving_executors.count(node)
        # note: anticipated_task_idx can be larger than node.num_tasks
        # when the tasks finish very fast before commitments are fulfilled
        return anticipated_task_idx >= node.num_tasks

    def schedule(self):
        executor = next(iter(self.exec_to_schedule))
        source = executor.job_dag if executor.node is None else executor.node

        # schedule executors from the source until the commitment is fulfilled
        while len(self.exec_commit[source]) > 0 and \
              len(self.exec_to_schedule) > 0:

            # keep fulfilling the commitment using free executors
            node = self.exec_commit.pop(source)
            executor = self.exec_to_schedule.pop()

            # mark executor as in use if it was free executor previously
            if self.free_executors.contain_executor(executor.job_dag, executor):
                self.free_executors.remove(executor)

            if node is None:
                # the next node is explicitly silent, make executor ilde
                if executor.job_dag is not None and \
                   any([not n.no_more_tasks for n in \
                        executor.job_dag.nodes]):
                    # mark executor as idle in its original job
                    self.free_executors.add(executor.job_dag, executor)
                else:
                    # no where to assign, put executor in null pool
                    self.free_executors.add(None, executor)


            elif not node.no_more_tasks:
                # node is not currently saturated
                if executor.job_dag == node.job_dag:
                    # executor local to the job
                    if node in node.job_dag.frontier_nodes:
                        # node is immediately runnable
                        task = node.schedule(executor)
                        self.timeline.push(task.finish_time, task)
                    else:
                        # put executor back in the free pool
                        self.free_executors.add(executor.job_dag, executor)

                else:
                    # need to move executor
                    self.timeline.push(
                        self.wall_time.curr_time + config.moving_delay, executor)
                    # keep track of moving executors
                    self.moving_executors.add(executor, node)

            else:
                # node is already saturated, use backup logic
                self.backup_schedule(executor)

    def step(self, action):

        assert self.action_space.contains(action)

        next_node, limit_idx = action

        # index starts from 0 but degree of parallelism starts with 1
        limit = limit_idx + 1

        # mark the node as selected
        assert next_node not in self.node_selected
        self.node_selected.add(next_node)
        # commit the source executor
        executor = next(iter(self.exec_to_schedule))
        source = executor.job_dag if executor.node is None else executor.node

        # compute number of valid executors to assign
        if next_node is not None:
            use_exec = min(next_node.num_tasks - next_node.next_task_idx - \
                           self.exec_commit.node_commit[next_node] - \
                           self.moving_executors.count(next_node), limit,
                           self.num_source_exec)
        else:
            use_exec = self.num_source_exec

        assert use_exec > 0

        self.exec_commit.add(source, next_node, use_exec)
        # deduct the executors that know the destination
        self.num_source_exec -= use_exec
        assert self.num_source_exec >= 0

        if self.num_source_exec == 0:
            # now a new scheduling round, clean up node selection
            self.node_selected.clear()
            # all commitments are made, now schedule free executors
            self.schedule()

        # Now run to the next event in the virtual timeline
        while len(self.timeline) > 0 and self.num_source_exec == 0:
            # consult agent by putting executors in source_exec

            new_time, obj = self.timeline.pop()
            self.wall_time.update_time(new_time)

            # case task: a task completion event, and frees up an executor.
            # case query: a new job arrives
            # case executor: an executor arrives at certain job

            if isinstance(obj, Task):  # task completion event
                finished_task = obj
                node = finished_task.node
                node.num_finished_tasks += 1

                # bookkeepings for node completion
                frontier_changed = False
                if node.num_finished_tasks == node.num_tasks:
                    assert not node.tasks_all_done  # only complete once
                    node.tasks_all_done = True
                    node.job_dag.num_nodes_done += 1
                    node.node_finish_time = self.wall_time.curr_time

                    frontier_changed = node.job_dag.update_frontier_nodes(node)

                # assign new destination for the job
                self.assign_executor(finished_task.executor, frontier_changed)

                # bookkeepings for job completion
                if node.job_dag.num_nodes_done == node.job_dag.num_nodes:
                    assert not node.job_dag.completed  # only complete once
                    node.job_dag.completed = True
                    node.job_dag.completion_time = self.wall_time.curr_time
                    self.remove_job(node.job_dag)

            elif isinstance(obj, JobDAG):  # new job arrival event
                job_dag = obj
                # job should be arrived at the first time
                assert not job_dag.arrived
                job_dag.arrived = True
                # inform agent about job arrival when stream is enabled
                self.job_dags.add(job_dag)
                self.add_job(job_dag)
                self.action_map = compute_act_map(self.job_dags)
                # assign free executors (if any) to the new job
                if len(self.free_executors[None]) > 0:
                    self.exec_to_schedule = \
                        OrderedSet(self.free_executors[None])
                    self.source_job = None
                    self.num_source_exec = \
                        len(self.free_executors[None])

            elif isinstance(obj, Executor):  # executor arrival event
                executor = obj
                # pop destination from the tracking record
                node = self.moving_executors.pop(executor)

                if node is not None:
                    # the job is not yet done when executor arrives
                    executor.job_dag = node.job_dag
                    node.job_dag.executors.add(executor)

                if node is not None and not node.no_more_tasks:
                    # the node is still schedulable
                    if node in node.job_dag.frontier_nodes:
                        # node is immediately runnable
                        task = node.schedule(executor)
                        self.timeline.push(task.finish_time, task)
                    else:
                        # free up the executor in this job
                        self.free_executors.add(executor.job_dag, executor)
                else:
                    # the node is saturated or the job is done
                    # by the time the executor arrives, use
                    # backup logic
                    self.backup_schedule(executor)

            else:
                print("illegal event type")
                exit(1)

        # compute reward
        reward = self.reward_calculator.get_reward(
            self.job_dags, self.wall_time.curr_time)

        # no more decision to make, jobs all done or time is up
        done = (self.num_source_exec == 0) and \
               ((len(self.timeline) == 0) or \
               (self.wall_time.curr_time >= self.max_time))

        if done:
            assert self.wall_time.curr_time >= self.max_time or \
                   len(self.job_dags) == 0

        return self.observe(), reward, done, None

    def remove_job(self, job_dag):
        for executor in list(job_dag.executors):
            executor.detach_job()
        self.exec_commit.remove_job(job_dag)
        self.free_executors.remove_job(job_dag)
        self.moving_executors.remove_job(job_dag)
        self.job_dags.remove(job_dag)
        self.finished_job_dags.add(job_dag)
        remove_job_from_graph(self.graph, job_dag)
        self.action_map = compute_act_map(self.job_dags)

    def reset(self, max_time=np.inf):
        # reset observation and action space
        self.setup_space()

        self.max_time = max_time
        self.wall_time.reset()
        self.timeline.reset()
        self.exec_commit.reset()
        self.moving_executors.reset()
        self.reward_calculator.reset()
        self.finished_job_dags = OrderedSet()
        self.node_selected.clear()
        for executor in self.executors:
            executor.reset()
        self.free_executors.reset(self.executors)
        # generate a set of new jobs
        self.job_dags = generate_jobs(
            self.np_random, self.timeline, self.wall_time)
        # map action to dag_idx and node_idx
        self.action_map = compute_act_map(self.job_dags)
        # add initial set of jobs in the system
        for job_dag in self.job_dags:
            self.add_job(job_dag)
        # put all executors as source executors initially
        self.source_job = None
        self.num_source_exec = len(self.executors)
        self.exec_to_schedule = OrderedSet(self.executors)

        return self.observe()

    def seed(self, seed):
        self.np_random = seeding.np_random(seed)

    def setup_space(self):
        # Set up the observation and action space
        # The boundary of the space may change if the dynamics is changed
        # a warning message will show up every time e.g., the observation falls
        # out of the observation space
        self.graph = DirectedGraph()
        self.obs_node_low = np.array([0] * 6)
        self.obs_node_high = np.array([config.exec_cap, 1, config.exec_cap, 1000, 100000, 1])
        self.obs_edge_low = self.obs_edge_high = np.array([])  # features on nodes only
        self.observation_space = spaces.Graph(
            node_feature_space=spaces.MultiBox(
                low=self.obs_node_low,
                high=self.obs_node_high,
                dtype=np.float32),
            edge_feature_space=spaces.MultiBox(
                low=self.obs_edge_low,
                high=self.obs_edge_high,
                dtype=np.float32))
        self.action_space = spaces.Tuple(
            (spaces.NodeInGraph(self.graph),
            spaces.MaskedDiscrete(config.num_servers)))

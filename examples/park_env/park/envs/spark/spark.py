import os
import sys
import zmq
import time
import numpy as np
import multiprocessing as mp
from datetime import datetime
from collections import OrderedDict

import park
from park import core, spaces, logger
from park.param import config
from park.utils import seeding
from park.utils.ordered_set import OrderedSet
from park.utils.directed_graph import DirectedGraph
from park.envs.spark.dags_database import DAGsDatabase
from park.envs.spark.environment import Environment
from park.envs.spark.executor_tracking import ExecutorMap
from park.envs.spark_sim.job_graph import add_job_in_graph, remove_job_from_graph

try:
    from park.envs.spark.ipc_msg_pb2 import IPCMessage, IPCReply
except:
    os.system("protoc -I=./park/envs/spark/ --python_out=./park/envs/spark/ ./park/envs/spark/ipc_msg.proto")
    from park.envs.spark.ipc_msg_pb2 import IPCMessage, IPCReply


class SparkEnv(core.SysEnv):
    """
    Interacting with a modified scheduling module in Apache Spark.
    See reference for more details.

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
        Section 6.1
        Learning Scheduling Algorithms for Data Processing Clusters
        H Mao, M Schwarzkopf, SB Venkatakrishnan, M Alizadeh
        https://arxiv.org/pdf/1810.01963.pdf
    """
    def __init__(self):
        # TODO: check if spark exists, download, build

        # random seed
        self.seed(config.seed)

    def run(self, agent_class, *args, **kwargs):

        # restart spark and scheduling module
        park_path = park.__path__[0]
        os.system("ps aux | grep -ie spark-tpch | awk '{print $2}' | xargs kill -9")
        os.system(park_path + '/envs/spark/spark/sbin/stop-master.sh')
        os.system(park_path + '/envs/spark/spark/sbin/stop-slaves.sh')
        os.system(park_path + '/envs/spark/spark/sbin/stop-shuffle-service.sh')
        os.system(park_path + '/envs/spark/spark/sbin/start-master.sh')
        os.system(park_path + '/envs/spark/spark/sbin/start-slave.sh')
        os.system(park_path + '/envs/spark/spark/sbin/start-shuffle-service.sh')

        # start the server process
        server_process = SchedulingServer(agent_class, args, kwargs)
        server_process.start()

        # TODO: sample a random set of jobs
        submit_script = 'python3 ' + park_path + '/envs/spark/submit_tpch.py'
        os.system(submit_script)

        server_process.join()

    def seed(self, seed):
        self.np_random = seeding.np_random(seed)


class SchedulingServer(mp.Process):
    def __init__(self, agent_class, *args, **kwargs):
        mp.Process.__init__(self)
        self.dag_db = DAGsDatabase()
        self.exec_tracker = ExecutorMap()
        self.env = Environment(self.dag_db)
        self.exit = mp.Event()

        # set up space
        self.graph = DirectedGraph()
        self.obs_node_low = np.array([0] * 6)
        self.obs_node_high = np.array([
            config.exec_cap, 1, config.exec_cap, 1000, 100000, 1])
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
        self.action_space = spaces.NodeInGraph(self.graph)

        # set up agent
        self.agent = agent_class(
            self.observation_space, self.action_space, args, kwargs)

    def run(self):

        # set up ipc communication
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        ipc_msg = IPCMessage()
        ipc_reply = IPCReply()

        os.system('rm /tmp/spark_scheduling_java_python_ipc')
        socket.bind("ipc:///tmp/spark_scheduling_java_python_ipc")

        # for reward computation
        num_active_jobs = 0
        prev_time = time.time()

        while not self.exit.is_set():
            msg = socket.recv()
            ipc_msg.ParseFromString(msg)

            if ipc_msg.msg_type == 'register':
                self.dag_db.add_new_app(ipc_msg.app_name, ipc_msg.app_id)
                job_dag = self.env.add_job_dag(ipc_msg.app_id)
                add_job_in_graph(self.graph, job_dag)
                ipc_reply.msg = \
                    "external scheduler register app " + str(ipc_msg.app_name)

            elif ipc_msg.msg_type == 'bind':
                self.env.bind_exec_id(ipc_msg.app_id, ipc_msg.exec_id, ipc_msg.track_id)
                ipc_reply.msg = \
                    "external scheduler bind app_id " + \
                    str(ipc_msg.app_id) + " exec_id " + \
                    str(ipc_msg.exec_id) + " on track_id " + \
                    str(ipc_msg.track_id)

            elif ipc_msg.msg_type == 'inform':
                self.env.complete_tasks(
                    ipc_msg.app_id, ipc_msg.stage_id, ipc_msg.num_tasks_left)
                ipc_reply.msg = \
                    "external scheduler updated app_id " + \
                    str(ipc_msg.app_id) + \
                    " stage_id " + \
                    str(ipc_msg.stage_id) + \
                    " with " + str(ipc_msg.num_tasks_left) + " tasks left"

            elif ipc_msg.msg_type == 'update':
                frontier_nodes_changed = \
                    self.env.complete_stage(ipc_msg.app_id, ipc_msg.stage_id)

                ipc_reply.msg = \
                    "external scheduler updated app_id " + \
                    str(ipc_msg.app_id) + \
                    " stage_id " + \
                    str(ipc_msg.stage_id)

            elif ipc_msg.msg_type == 'tracking':
                # master asks which app it should assign the executor to
                ipc_reply.app_id, ipc_reply.num_executors_to_take = \
                    self.exec_tracker.pop_executor_flow(ipc_msg.num_available_executors)
                ipc_reply.msg = \
                    "external scheduler moves " + \
                    str(ipc_reply.num_executors_to_take) + \
                    " executor to app " + ipc_reply.app_id

            elif ipc_msg.msg_type == 'consult':

                # convert ipc_msg.app_id and ipc_msg.stage_id to corresponding
                # executors in virtual environment and then inovke the
                # scheduling agent

                # 1. translate the raw information into observation space
                # sort out the exec_map (where the executors are)
                exec_map = {job_dag: 0 for job_dag in self.env.job_dags}
                for app_id in self.dag_db.apps_map:
                    if app_id in self.exec_tracker.executor_flow:
                        job_dag = self.dag_db.apps_map[app_id]
                        exec_map[job_dag] = self.exec_tracker.executor_flow[app_id]

                source_job = self.dag_db.apps_map[ipc_msg.app_id]

                frontier_nodes = OrderedSet()
                for job_dag in self.env.job_dags:
                    for node in job_dag.frontier_nodes:
                        frontier_nodes.add(node)

                for job_dag in self.env.job_dags:
                    for node in job_dag.nodes:
                        feature = np.zeros([6])
                        # number of executors already in the job
                        feature[0] = exec_map[job_dag]
                        # source executor is from the current job (locality)
                        feature[1] = job_dag is source_job
                        # number of source executors
                        feature[2] = 1
                        # remaining number of tasks in the node
                        feature[3] = node.num_tasks - node.next_task_idx
                        # average task duration of the node
                        feature[4] = node.tasks[-1].duration
                        # is the current node valid
                        feature[5] = node in frontier_nodes

                        # update feature in observation
                        self.graph.update_nodes({node: feature})

                # update mask in the action space
                self.action_space.update_valid_set(frontier_nodes)

                # 2. gather feedback for the previous action
                curr_time = time.time()
                elapsed_time = curr_time - prev_time
                prev_reward = num_active_jobs * elapsed_time
                prev_done = False  # spark can be long running
                prev_info = {'elapsed_time': elapsed_time}
                num_active_jobs = len(self.env.job_dags)
                prev_time = curr_time

                # 3. get the action from the agent
                node = self.agent.get_action(self.graph, prev_reward, prev_done, prev_info)

                # 4. translate the action to ipc reply
                if node is None:
                    # no-action was made
                    ipc_reply.app_id = 'void'
                    ipc_reply.stage_id = -1
                else:
                    ipc_reply.app_id, ipc_reply.stage_id = self.env.spark_inverse_node_map[node]
                    if node.idx not in node.job_dag.frontier_nodes:
                        # move (or stay) the executor to the job only
                        ipc_reply.stage_id = -1

                if ipc_msg.app_id != 'void' and \
                   ipc_reply.app_id != 'void' and \
                   ipc_msg.app_id != ipc_reply.app_id:
                    # executor needs to move to another job, keep track of it
                    self.exec_tracker.add_executor_flow(ipc_reply.app_id, 1)

                ipc_reply.msg = \
                    "external scheduler return app_id " + str(ipc_reply.app_id) + \
                    " stage_id " + str(ipc_reply.stage_id) + \
                    " for exec_id " + str(ipc_msg.exec_id)

            elif ipc_msg.msg_type == 'deregister':
                job_dag = self.env.remove_job_dag(ipc_msg.app_id)
                remove_job_from_graph(self.graph, job_dag)
                self.dag_db.remove_app(ipc_msg.app_id)
                self.exec_tracker.remove_app(ipc_msg.app_id)
                ipc_reply.msg = \
                    "external scheduler deregister app " + ipc_msg.app_id

            print("time:", datetime.now())
            print("msg_type:", ipc_msg.msg_type)
            print("app_name:", ipc_msg.app_name)
            print("app_id:", ipc_msg.app_id)
            print("stage_id:", ipc_msg.stage_id)
            print("executor_id:", ipc_msg.exec_id)
            print("track_id:", ipc_msg.track_id)
            print("num_available_executors:", ipc_msg.num_available_executors)
            print("num_tasks_left", ipc_msg.num_tasks_left)
            print("reply_msg:", ipc_reply.msg)
            print("")
            sys.stdout.flush()

            socket.send(ipc_reply.SerializeToString())

    def shutdown(self):
        self.exit.set()



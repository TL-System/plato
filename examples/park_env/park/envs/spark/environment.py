from park.envs.spark_sim.executor import *
from park.envs.spark_sim.job_dag import *
from park.param import config
from park.utils.ordered_set import OrderedSet


class Environment(object):
    def __init__(self, dag_db):

        self.dag_db = dag_db

        self.job_dags = OrderedSet()
        self.action_map = {}  # action index -> node
        self.available_executors = {}
        self.last_trigger = None

        # executors
        self.executors = {}
        for exec_id in range(config.exec_cap):
            self.executors[exec_id] = Executor(exec_id)

        # dynamically bind {app_id -> job_dag}
        self.spark_dag_map = {}
        # dynamically bind {job_dag -> app_id}
        self.spark_inverse_dag_map = {}
        # dynamically bind {(app_id, stage_id) -> node}
        self.spark_node_map = {}
        # dynamically bind {node -> (app_id, stage_id)}
        self.spark_inverse_node_map = {}

        # dynamically bind {app_id -> {exec_id -> re-usable track_id}}
        self.exec_id_track_id_map = {}

    def add_job_dag(self, app_id):
        job_dag = self.dag_db.apps_map[app_id]
        job_dag.arrived = True

        self.job_dags.add(job_dag)

        # update map for job_dag
        self.spark_dag_map[app_id] = job_dag
        self.spark_inverse_dag_map[job_dag] = app_id

        # update exec_id track_id bind map
        self.exec_id_track_id_map[app_id] = {}

        # update map for node
        node_idx_to_stage_id_map = self.dag_db.stage_map[app_id]
        for node in job_dag.nodes:
            stage_id = node_idx_to_stage_id_map[node.idx]
            self.spark_node_map[(app_id, stage_id)] = node
            self.spark_inverse_node_map[node] = (app_id, stage_id)

        # update map for actions
        self.action_map.clear()
        self.action_map.update(self.pre_compute_action_map())

        return job_dag

    def bind_exec_id(self, app_id, exec_id, track_id):
        assert 0 <= track_id < config.exec_cap
        self.exec_id_track_id_map[app_id][exec_id] = track_id

    def complete_stage(self, app_id, stage_id):
        node = self.spark_node_map[(app_id, stage_id)]
        # bookkeepings for node completion
        assert not node.tasks_all_done  # only complete once
        node.tasks_all_done = True
        node.job_dag.update_frontier_nodes(node)
        node.job_dag.num_nodes_done += 1

        # bookkeepings for job completion
        if node.job_dag.num_nodes_done == node.job_dag.num_nodes:
            assert not node.job_dag.completed  # only complete once
            node.job_dag.completed = True

    def complete_tasks(self, app_id, stage_id, num_tasks_left):
        node = self.spark_node_map[(app_id, stage_id)]
        prev_finished_tasks = node.num_finished_tasks
        # update number of finished tasks for the node
        node.num_finished_tasks = node.num_tasks - num_tasks_left
        # update the next task index of the node
        node.next_task_idx += node.num_finished_tasks - prev_finished_tasks
        # remove node from frontier node if it is saturated
        node.no_more_tasks = (node.next_task_idx >= node.num_tasks)
        if node.no_more_tasks:
            if node.idx in node.job_dag.frontier_nodes:
                del node.job_dag.frontier_nodes[node.idx]

    def pre_compute_action_map(self):
        # translate action ~ [0, num_nodes_in_all_dags) to node object
        action_map = {}
        action = 0
        for job_dag in self.job_dags:
            for node in job_dag.nodes:
                action_map[action] = node
                action += 1
        return action_map

    def remove_job_dag(self, app_id):
        job_dag = self.dag_db.apps_map[app_id]

        self.job_dags.remove(job_dag)

        # free up stage holding executors
        for executor in job_dag.executors:
            executor.task = None
            executor.job_dag = None

        # update exec_id track_id map
        del self.exec_id_track_id_map[app_id]

        # update map for job_dag
        del self.spark_dag_map[app_id]
        del self.spark_inverse_dag_map[job_dag]

        # update map for node
        node_idx_to_stage_id_map = self.dag_db.stage_map[app_id]
        for node in job_dag.nodes:
            stage_id = node_idx_to_stage_id_map[node.idx]
            del self.spark_node_map[(app_id, stage_id)]
            del self.spark_inverse_node_map[node]

        # update map for actions
        self.action_map.clear()
        self.action_map.update(self.pre_compute_action_map())

        return job_dag

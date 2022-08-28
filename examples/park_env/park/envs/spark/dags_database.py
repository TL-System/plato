import os
import wget
import copy
import park
import zipfile
import numpy as np
from park.envs.spark_sim.wall_time import WallTime
from park.envs.spark_sim.job_dag import merge_job_dags
from park.envs.spark_sim.job_generator import load_job


class DAGsDatabase(object):
    """
    Map spark application id to known DAG
    """
    def __init__(self):
        # dummy wall_time
        self.wall_time = WallTime()
        # stores app_name -> job_dag (can be multiple DAGs)
        self.apps_store = {}
        # stores app_name -> {node.idx -> stage_id} map
        self.stage_store = {}

        # dynamically bind app_id -> job_dag
        self.apps_map = {}
        # dynamically bind app_id -> {node.idx -> stage_id}
        self.stage_map = {}
        # dynamically bind app_id -> app_name (query)
        self.queries_map = {}

        # number of tpch queries
        tpch_num = 22
        # tpch sizes
        tpch_size = ['2g','5g','10g','20g','50g','80g','100g']
        # tpch trace folder
        tpch_folder = park.__path__[0] + '/envs/spark/tpch/'
        # dummy np_random
        np_random = np.random.RandomState()

        # download tpch folder if not existed
        if not os.path.exists(park.__path__[0] + '/envs/spark/tpch/'):
            wget.download(
                'https://www.dropbox.com/s/w4xha9rf92851vy/tpch.zip?dl=1',
                out=park.__path__[0] + '/envs/spark/')
            with zipfile.ZipFile(
                 park.__path__[0] + '/envs/spark/tpch.zip', 'r') as zip_f:
                zip_f.extractall(park.__path__[0] + '/envs/spark/')

        # initialize dags_store
        for query_size in tpch_size:
            for query_idx in range(1, tpch_num + 1):

                job_dag = load_job(
                    query_size, query_idx, self.wall_time, np_random)

                self.apps_store['tpch-' + query_size + \
                    '-' + str(query_idx)] = job_dag

                # load stage_id -> node_idx map
                stage_id_to_node_idx_map = \
                    np.load(tpch_folder + query_size + '/' + \
                        'stage_id_to_node_idx_map_' + \
                        str(query_idx) + '.npy').item()

                # build up the new map based on merged job_dag
                node_idx_to_stage_id_map = \
                    {stage_id_to_node_idx_map[k]: k for k in stage_id_to_node_idx_map}

                # store the {node.idx -> stage_id} map
                self.stage_store[
                    'tpch-' + query_size + '-' + str(query_idx)] = \
                    node_idx_to_stage_id_map

    def add_new_app(self, app_name, app_id):
        job_dag = None
        stage_map = None

        if app_name in self.apps_store:
            job_dag = copy.deepcopy(self.apps_store[app_name])
            stage_map = self.stage_store[app_name]

        self.apps_map[app_id] = job_dag
        self.stage_map[app_id] = stage_map
        self.queries_map[app_id] = app_name

    def remove_app(self, app_id):
        if app_id in self.apps_map:
            del self.apps_map[app_id]
            del self.stage_map[app_id]
            del self.queries_map[app_id]

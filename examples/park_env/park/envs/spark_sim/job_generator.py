import os
import wget
import zipfile
import park
import numpy as np
from park.param import config
from park.utils.ordered_set import OrderedSet
from park.envs.spark_sim.task import Task
from park.envs.spark_sim.node import Node
from park.envs.spark_sim.job_dag import JobDAG
from park.envs.spark_sim.set_with_count import SetWithCount


def load_job(query_size, query_idx, wall_time, np_random):
    file_path = park.__path__[0] + '/envs/spark_sim/traces/'
    if not os.path.exists(file_path):
        wget.download(
                'https://www.dropbox.com/s/pmoivv5eg9ldmfn/traces.zip?dl=1',
                out=park.__path__[0] + '/envs/spark_sim/')
        with zipfile.ZipFile(
             park.__path__[0] + '/envs/spark_sim/traces.zip', 'r') as zip_f:
            zip_f.extractall(park.__path__[0] + '/envs/spark_sim/')

    query_path = file_path + query_size + '/'
    
    adj_mat = np.load(
        query_path + 'adj_mat_' + str(query_idx) + '.npy')
    task_durations = np.load(
        query_path + 'task_duration_' + str(query_idx) + '.npy').item()
    
    assert adj_mat.shape[0] == adj_mat.shape[1]
    assert adj_mat.shape[0] == len(task_durations)

    num_nodes = adj_mat.shape[0]
    nodes = []
    for n in range(num_nodes):
        task_duration = task_durations[n]
        e = next(iter(task_duration['first_wave']))

        num_tasks = len(task_duration['first_wave'][e]) + \
                    len(task_duration['rest_wave'][e])

        # remove fresh duration from first wave duration
        # drag nearest neighbor first wave duration to empty spots
        pre_process_task_duration(task_duration)
        rough_duration = np.mean(
            [i for l in task_duration['first_wave'].values() for i in l] + \
            [i for l in task_duration['rest_wave'].values() for i in l] + \
            [i for l in task_duration['fresh_durations'].values() for i in l])

        # generate tasks in a node
        tasks = []
        for j in range(num_tasks):
            task = Task(j, rough_duration, wall_time)
            tasks.append(task)

        # generate a node
        node = Node(n, tasks, task_duration, wall_time, np_random)
        nodes.append(node)

    # parent and child node info
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_mat[i, j] == 1:
                nodes[i].child_nodes.append(nodes[j])
                nodes[j].parent_nodes.append(nodes[i])

    # initialize descendant nodes
    for node in nodes:
        if len(node.parent_nodes) == 0:  # root
            node.descendant_nodes = recursive_find_descendant(node)

    # generate DAG
    job_dag = JobDAG(nodes, adj_mat,
         'tpch-' + query_size + '-' + str(query_idx))

    return job_dag


def pre_process_task_duration(task_duration):
    # remove fresh durations from first wave
    clean_first_wave = {}
    for e in task_duration['first_wave']:
        clean_first_wave[e] = []
        fresh_durations = SetWithCount()
        # O(1) access
        for d in task_duration['fresh_durations'][e]:
            fresh_durations.add(d)
        for d in task_duration['first_wave'][e]:
            if d not in fresh_durations:
                clean_first_wave[e].append(d)
            else:
                # prevent duplicated fresh duration blocking first wave
                fresh_durations.remove(d)

    # fill in nearest neighour first wave
    last_first_wave = []
    for e in sorted(clean_first_wave.keys()):
        if len(clean_first_wave[e]) == 0:
            clean_first_wave[e] = last_first_wave
        last_first_wave = clean_first_wave[e]

    # swap the first wave with fresh durations removed
    task_duration['first_wave'] = clean_first_wave


def recursive_find_descendant(node):
    if len(node.descendant_nodes) > 0:  # already visited
        return node.descendant_nodes
    else:
        node.descendant_nodes = [node]
        for child_node in node.child_nodes:  # terminate on leaves automatically
            child_descendant_nodes = recursive_find_descendant(child_node)
            for dn in child_descendant_nodes:
                if dn not in node.descendant_nodes:  # remove dual path duplicates
                    node.descendant_nodes.append(dn)
        return node.descendant_nodes


def generate_jobs(np_random, timeline, wall_time):

    job_dags = OrderedSet()
    tpch_size = ['2g','5g','10g','20g','50g','80g','100g']
    tpch_num = 22
    t = 0

    for _ in range(config.num_init_dags):
        # generate query
        query_size = tpch_size[np_random.randint(len(tpch_size))]
        query_idx = str(np_random.randint(tpch_num) + 1)
        # generate job
        job_dag = load_job(
            query_size, query_idx, wall_time, np_random)
        # job already arrived, put in job_dags
        job_dag.start_time = t
        job_dag.arrived = True
        job_dags.add(job_dag)

    for _ in range(config.num_stream_dags):
        # poisson process
        t += int(np_random.exponential(config.stream_interval))
        # uniform distribution
        query_size = tpch_size[np_random.randint(len(tpch_size))]
        query_idx = str(np_random.randint(tpch_num) + 1)
        # generate job
        job_dag = load_job(
            query_size, query_idx, wall_time, np_random)
        # push into timeline
        job_dag.start_time = t
        timeline.push(t, job_dag)

    return job_dags

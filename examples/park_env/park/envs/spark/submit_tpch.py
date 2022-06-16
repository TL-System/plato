import os
import sys
import time
import subprocess
import numpy as np
import multiprocessing as mp
from HTMLParser import HTMLParser

import park


SPARK_BIN_PATH = park.__path__[0] + '/park/envs/spark/bin/'
SCHEDULER_MODEL = 'external'
TOTAL_EXEC_CORES = 50
MASTER_URL = 'spark://lab.local:7077'
JZMQ_PATH = '/usr/local/lib/'
CHECKPOINT_PREFIX = '/tmp/tpch_checkpoint_'
CHECK_INTERVAL = 1
EXP_TIMEOUT = 10000
SPARK_SLACK = 1


def run(query_size, query_idx, query_order, port):

    # query_size: size of input (e.g., '2g', '50g')
    # query_idx: index of tpc-h query, starting from 1 (e.g., 3, 21)
    # query_order: index of the query in multiple submission (e.g., 0, 1)

    out = open('/tmp/tpch' + '_out_size_' + query_size + '_q_' + \
               str(query_idx) + '_idx_' + str(query_order), 'w')
    err = open('/tmp/tpch' + '_err_size_' + query_size + '_q_' + \
               str(query_idx) + '_idx_' + str(query_order), 'w')

    subprocess.call(
        [SPARK_BIN_PATH + 'spark-submit' +
         ' --class "main.scala.TpchQuery"' +
         ' --conf spark.shuffle.service.enabled=true' +
         ' --conf spark.locality.wait=0s' +
         ' --conf spark.scheduler.type=' + SCHEDULER_MODEL +
         ' --conf spark.sql.broadcastTimeout=1200' +
         ' --conf spark.ui.port=' + str(port) +
         ' --executor-cores 1' +
         ' --total-executor-cores ' + str(TOTAL_EXEC_CORES) +
         ' --name "tpch-' + query_size + '-' + str(query_idx) + '"' +
         ' --master ' + MASTER_URL +
         ' --driver-java-options ' +
         '"-Djava.library.path='+ JZMQ_PATH +'"' +
         ' tpch-hdfs/spark-tpch-' + query_size + '.jar ' +
         str(query_idx) + ' ' + str(query_order)],
         stdout=out, stderr=err, shell=True)

    out.close()
    err.close()


def main():

    port = 4040

    queries = sys.argv[1]

    for query_order in xrange(len(queries)):

        query = queries[query_order]

        query_parse = query.split('-')
        query_size = query_parse[1]
        query_idx = query_parse[2]

        os.system('rm ' + CHECKPOINT_PREFIX + str(query_idx) + '_' + str(query_order))

        p = mp.Process(target=run, args=(query_size, query_idx, query_order, port))
        p.start()

        port += 1

    start_time = time.time()

    # check results of all running queries
    running_queries = {}
    for query_order in xrange(len(queries)):
        running_queries[query_order] = queries[query_order]

    while len(running_queries) > 0:

        time.sleep(CHECK_INTERVAL)

        for query_order in running_queries.keys():

            query = running_queries[query_order]

            query_parse = query.split('-')
            query_idx = query_parse[2]

            if os.path.isfile(CHECKPOINT_PREFIX + str(query_idx) + '_' + str(query_order)):

                # remove current query from running queries dict
                del running_queries[query_order]

        if time.time() - start_time >= EXP_TIMEOUT:
            print("Experiment timeout")
            exit(1)  # // timeout

    # slack
    time.sleep(SPARK_SLACK)


if __name__ == '__main__':
    main()
import park

class Params:
    #### Parameters that you may change #####
    
    # The number of queries to run in each step.
    QUERIES_PER_STEP = 1000
    
    # The number of steps (i.e. query workloads) to run before declaring the epoch as done.
    STEPS_PER_EPOCH = 10
    
    # Options are 'full' or 'sampled'
    DATASET_TYPE = 'full'
    
    
    
    #### Parameters you should NOT change ####
    
    # This is dependent on the dataset being used. The OSM dataset has 6 attributes.
    NDIMS = 6
    print(park.__path__[0])    
    # The name of the datafile. If not present on the machine, it will be downloaded.
    assert DATASET_TYPE in ['full']
    DATASET_PATH = park.__path__[0] + '/envs/multi_dim_index/data/osm_dataset_%s.bin' % DATASET_TYPE
    DATA_DOWNLOAD_URL = 'https://www.dropbox.com/s/ws9o25cfc4znvc7/us_northeast_osm.bin?dl=1'
    DATA_SUMMARY_DIR = park.__path__[0] + '/envs/multi_dim_index/data'

    BINARY_PATH = park.__path__[0] + '/envs/multi_dim_index/exec/mdi_db'
    BINARY_DOWNLOAD_URL = 'https://www.dropbox.com/s/5w1ngr2ejnxso0z/mdi_db?dl=1'


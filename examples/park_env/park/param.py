import argparse

parser = argparse.ArgumentParser(description='parameters')

# -- Basic --
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
parser.add_argument('--eps', type=float, default=1e-6,
                    help='epsilon (default: 1e-6)')
parser.add_argument('--logging_level', type=str, default='info',
                    help='logging level (default: info)')
parser.add_argument('--log_to', type=str, default='print',
                    help='logging destination, "print" or a filepath (default: print)')

# -- Load balance --
parser.add_argument('--num_servers', type=int, default=10,
                    help='number of servers (default: 10)')
parser.add_argument('--num_stream_jobs', type=int, default=1000,
                    help='number of streaming jobs (default: 1000)')
parser.add_argument('--service_rates', type=float,
                    default=[0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05],
                    nargs='+', help='workers service rates '
                    '(default: [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05])')
parser.add_argument('--job_interval', type=int, default=55,
                    help='job arrival interval (default: 55)')
parser.add_argument('--job_size_pareto_shape', type=float, default=1.5,
                    help='pareto job size distribution shape (default: 1.5)')
parser.add_argument('--job_size_pareto_scale', type=float, default=100.0,
                    help='pareto job size distribution scale (default: 100.0)')
parser.add_argument('--load_balance_obs_high', type=float, default=500000.0,
                    help='observation cap for load balance env (default: 500000.0)')

# -- AQM --
parser.add_argument('--aqm_link_delay', type=int, default=10,
                    help='mahimahi link delay in millisecond (default: 10)')
parser.add_argument('--aqm_step_num', type=int, default=300,
                    help='total number of steps (default: 300)')
parser.add_argument('--aqm_step_interval', type=int, default=100,
                    help='time interval of each step in millisecond (default: 100)')
parser.add_argument('--aqm_uplink_trace', type=str, default="park/envs/aqm/mahimahi/trace10",
                    help='mahimahi uplink trace file')
parser.add_argument('--aqm_downlink_trace', type=str, default="park/envs/aqm/mahimahi/trace10",
                    help='mahimahi downlink trace file')

# -- Spark --
parser.add_argument('--exec_cap', type=int, default=50,
                    help='Number of total executors (default: 50)')
parser.add_argument('--num_init_dags', type=int, default=20,
                    help='Number of initial DAGs in system (default: 20)')
parser.add_argument('--num_stream_dags', type=int, default=100,
                    help='number of streaming DAGs (default: 100)')
parser.add_argument('--stream_interval', type=int, default=25000,
                    help='inter job arrival time in milliseconds (default: 25000)')
parser.add_argument('--moving_delay', type=int, default=2000,
                    help='Moving delay (milliseconds) (default: 2000)')
parser.add_argument('--warmup_delay', type=int, default=1000,
                    help='Executor warming up delay (milliseconds) (default: 1000)')

# -- Query Optimizer --
parser.add_argument('--qopt_java_output', action="store_false",
                    help="should the java servers output be visible")
# TODO: need to add more control for this
parser.add_argument('--qopt_viz', type=int, default=0,
                    help="visualizations per episode")

# parameters to be passed to the calcite backend
parser.add_argument('--qopt_eval_runtime', type=int, default=0,
                    help="execute query plan on db, to get runtimes, when evaluating")
parser.add_argument('--qopt_train_runtime', type=int, default=0,
                    help="train using runtimes from DB")

parser.add_argument('--qopt_port', type=int, default=2654,
                    help="port for communicaton with calcite backend")
parser.add_argument('--qopt_query', type=int, default=0,
                    help="index of the query to run")
parser.add_argument('--qopt_train', type=int, default=1,
                    help="""0 or 1. To run in training mode or test mode. Check
                    the calcite backend for more description of the two
                    modes.""")
parser.add_argument('--qopt_only_final_reward', type=int,default=0, 
                    help="""0 or 1. If true, then only the final reward will be
                    returned.""")
parser.add_argument('--qopt_lopt', type=int,default=0, help="0 or 1")
parser.add_argument('--qopt_exh', type=int,default=0, help="0 or 1")
parser.add_argument('--qopt_verbose', type=int,default=0, help="0 or 1")
parser.add_argument('--qopt_left_deep', type=int,default=0, help="0 or 1")
parser.add_argument('--qopt_only_attr_features', type=int, required=False,
                            default=1, help='')

parser.add_argument('--qopt_reward_normalization', type=str, required=False,
                            default='min_max', help='type of reward normalization')
parser.add_argument('--qopt_cost_model', type=str, required=False,
                            default='rowCount', help='')
parser.add_argument('--qopt_dataset', type=str, required=False,
                            default='JOB', help='')
parser.add_argument('--qopt_clear_cache', type=int, required=False,
                            default=0, help='')
parser.add_argument('--qopt_recompute_fixed_planners', type=int, required=False,
                            default=0, help='')

# -- Cache --
parser.add_argument('--cache_trace', type=str, required=False, default='test', 
                    help='trace selection')
parser.add_argument('--cache_size', type=int, required=False, default=1024, 
                    help='size of network cache')
parser.add_argument('--cache_unseen_recency', type=int, required=False, default=500,
                    help='default number for the recency feature')

# -- Simple Queue --
parser.add_argument('--sq_num_servers', type=float, default=5,
                    help='Number of server in simple queue environment (default: 5)')
parser.add_argument('--sq_free_up_prob', type=float, default=0.5,
                    help='Probability for a server to free up (default: 0.5)')

# -- Switch Scheduling --
parser.add_argument('--ss_num_ports', type=int, default=3,
                     help='Number of ports (same for input and output) (default: 3)')
parser.add_argument('--ss_state_max_queue', type=int, default=50,
                     help='Max queue size in state before clipping (default: 50)')
parser.add_argument('--ss_load', type=float, default=0.9,
                     help='Load of the system (default: 0.9)')

# -- Device Placement for Tensorflow --
parser.add_argument('--pl_graph', type=str, default='inception',
                    help='The tensorflow graph to place')
parser.add_argument('--pl_n_devs', type=int, default=2,
                    help='Number of devices to split the graph across')

# -- Circuit Simulator --
parser.add_argument('--circuit_remote_host', type=str, default=None,
                    help='The remote host of circuit simulation server (default: None)')
parser.add_argument('--circuit_remote_port', type=int, default=None,
                    help='The remote port of circuit simulation server (default: None)')
parser.add_argument('--circuit_tmp_path', type=str, default='./tmp',
                    help='The temporary path to the simulator (default: ./tmp)')
parser.add_argument('--circuit_env_type', type=str, default='pointed',
                    help='The circuit environment type (default: pointed)')
parser.add_argument('--circuit_total_steps', type=int, default=5,
                    help='The total steps of the environment (default: 5)')

# -- Congestion Control --
parser.add_argument('--cc_delay', type=int, default=25, help='Link delay to run experiment with')
parser.add_argument('--cc_uplink_trace', type=str, default="const48.mahi", help='Uplink trace to use')
parser.add_argument('--cc_downlink_trace', type=str, default="const48.mahi", help='Uplink trace to use')
parser.add_argument('--cc_duration', type=int, default=120, help='How long of an experiment to run')

# -- Region Assignment
parser.add_argument('--ra_shuffle', type=bool, default=True, 
    help="Whether or not to shuffle the order that pages are assigned, or to use the creation order.")



config, _ = parser.parse_known_args()

# Format follows OpenAI gym https://gym.openai.com

from park.envs.registration import register, make

register(
    env_id='load_balance',
    entry_point='park.envs.load_balance:LoadBalanceEnv',
)

register(
    env_id='abr',
    entry_point='park.envs.abr:ABREnv',
)

register(
    env_id='abr_sim',
    entry_point='park.envs.abr_sim:ABRSimEnv',
)

register(
    env_id='aqm',
    entry_point='park.envs.aqm:AQMEnv',
)

register(
    env_id='congestion_control',
    entry_point='park.envs.congestion_control:CongestionControlEnv',
)

# register(
#     env_id='spark',  # under maintenance
#     entry_point='park.envs.spark:SparkEnv',
# )

register(
    env_id='spark_sim',
    entry_point='park.envs.spark_sim:SparkSimEnv',
)

register(
    env_id='query_optimizer',
    entry_point='park.envs.query_optimizer:QueryOptEnv',
)

register(
    env_id='cache',
    entry_point='park.envs.cache:CacheEnv',
)

register(
    env_id='simple_queue',
    entry_point='park.envs.simple_queue:SimpleQueueEnv',
)

register(
    env_id='switch_scheduling',
    entry_point='park.envs.switch_scheduling:SwitchEnv',
)

register(
    env_id='tf_placement',
    entry_point='park.envs.tf_placement:TFPlacementEnv'
)

register(
    env_id='tf_placement_sim',
    entry_point='park.envs.tf_placement_sim:TFPlacementSimEnv'
)

register(
    env_id='circuit_three_stage_transimpedance',
    entry_point='park.envs.circuit.entries:make_three_stage_transimpedance_amplifier_environment'
)

register(
    env_id='region_assignment',
    entry_point='park.envs.region_assignment:RegionAssignmentEnv'
)

register(
    env_id='multi_dim_index',
    entry_point='park.envs.multi_dim_index:MultiDimIndexEnv'
)


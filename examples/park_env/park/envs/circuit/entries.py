from park.envs.circuit.environment import CircuitPointedEnv
from park.envs.circuit.simulator.circuit import ThreeStageTranimpedenceAmplifier, RemoteContext, LocalContext
from park.param import config


def make_three_stage_transimpedance_amplifier_environment():
    from park.envs.circuit.env_config.transimpedance.three_stage import \
        benchmark, obs_mark, lower_bound, upper_bound, preset

    if config.circuit_remote_host is not None and config.circuit_remote_port is not None:
        context = RemoteContext(config.circuit_remote_host, config.circuit_remote_port)
    else:
        context = LocalContext(config.circuit_tmp_path)

    evaluator = ThreeStageTranimpedenceAmplifier(default_context=context).evaluator()
    for key in evaluator.parameters:
        if key not in preset:
            evaluator.set_bound(key, lower_bound[key], upper_bound[key])
    for key in preset:
        setattr(evaluator, key, preset[key])

    if config.circuit_env_type == 'pointed':
        env = CircuitPointedEnv(evaluator, benchmark, obs_mark, config.circuit_total_steps)
    else:
        raise ValueError(f'Cannot found environment wrapper "{config.circuit_env_type}"')

    return env

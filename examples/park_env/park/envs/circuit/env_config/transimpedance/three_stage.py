import functools

from park.envs.circuit.simulator.circuit import Rater

rater = Rater(-5) \
    .metric('gain', scale='log', direction='maximize', constrained=True, targeted=False) \
    .metric('bandwidth', scale='log', direction='maximize', constrained=True, targeted=False) \
    .metric('power', scale='linear', direction='minimize', constrained=0.003, targeted=True) \
    .metric('area', scale='linear', direction='minimize', constrained=False, targeted=True)

specs = {
    'bandwidth': 90.107e6,
    'power': 0.0013694,
    'gain_bandwidth': 1,
    'area': 211.2,
    'gain': 20236
}

benchmark = functools.partial(rater, specs=specs)

obs_mark = {}

preset = {
    'L1': 1,
    'LL1': 1,
    'LB1': 2,
    'L2': 1,
    'LL2': 1,
    'LB2': 2,
    'L3': 1,
    'LB3': 2,
    'LB': 2,
}

lower_bound = {
    'W1': 2,
    'WL1': 2,
    'WB1': 2,
    'W2': 2,
    'WL2': 2,
    'WB2': 2,
    'W3': 2,
    'WB3': 2,
    'WB': 2,
    'RB': 100
}

upper_bound = {
    'W1': 15,
    'WL1': 5,
    'WB1': 5,
    'W2': 5,
    'WL2': 5,
    'WB2': 10,
    'W3': 50,
    'WB3': 15,
    'WB': 5,
    'RB': 400
}

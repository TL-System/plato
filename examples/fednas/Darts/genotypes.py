from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

DARTS = DARTS_V2

FL_1 = Genotype(normal=[('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('sep_conv_5x5', 0), ('max_pool_3x3', 3), ('dil_conv_5x5', 0), ('skip_connect', 4)], reduce_concat=range(2, 6))

# glace 3000, total 6000, client 10
RL1 =  Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

# warmup 10k, search 6k, lr=3e-3, client 10, 97.02
RL2 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('skip_connect', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2), ('avg_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))

# warmup 10k, search 6k, lr=3e-3, client 10, v2, 96.0
RL3 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('sep_conv_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))

# warmup 10k, search 6k, lr=3e-3, client 10, v3, 96.16
RL4 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 2), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))

# warmup 10k, search 6k, lr=3e-3, client 10, v4, 96.96
RL5 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 4), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))

# warmup 10k, search 6k, lr=3e-3, client 10, v5, 96.92
RL6 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 2), ('dil_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))

# warmup 10k, search 6k, lr=3e-3, client 10, seed 2 96.25
RL7 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('skip_connect', 2), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2), ('avg_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))

# 96.07
RL8 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('max_pool_3x3', 2), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))

# 96.11
RL9 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('avg_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))

# 96.81
RL10 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('max_pool_3x3', 2), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 2), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))

# seed 1345 96.56
RL11 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('skip_connect', 2), ('skip_connect', 1), ('avg_pool_3x3', 3), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 3), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))

# 96.06
RL12 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 2), ('dil_conv_5x5', 3), ('skip_connect', 2), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))

RL13 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('max_pool_3x3', 2), ('sep_conv_3x3', 3), ('max_pool_3x3', 2), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 3), ('avg_pool_3x3', 0), ('avg_pool_3x3', 3)], reduce_concat=range(2, 6))

# warmup 10k, search 6k, lr=3e-3, client 10, seed 5, acc 97.38 *BEST
RL14 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 2), ('sep_conv_3x3', 3), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6))

C20 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))
C50 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 1), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))



# 96.77
SP = Genotype(normal=[('skip_connect', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))


DEEPRL = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))

NONIID = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3), ('sep_conv_5x5', 4), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 3), ('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))


#soft 
zero=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 4), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 2), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))
stale=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 1), ('sep_conv_5x5', 3), ('sep_conv_3x3', 4), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 3), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
throw=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('dil_conv_3x3', 4)], reduce_concat=range(2, 6))
use=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('max_pool_3x3', 2), ('skip_connect', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_5x5', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))

stale0=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('skip_connect', 2), ('max_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 4)], reduce_concat=range(2, 6))
stale2=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 2), ('dil_conv_5x5', 3), ('sep_conv_5x5', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 1), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))
stale3=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 2), ('sep_conv_3x3', 3), ('max_pool_3x3', 2), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 3), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))
stale4=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('skip_connect', 2), ('sep_conv_3x3', 3), ('max_pool_3x3', 2), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 1), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))
stale5=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 4), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 2), ('sep_conv_5x5', 3), ('max_pool_3x3', 1), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
stale6=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 2), ('dil_conv_5x5', 3), ('sep_conv_5x5', 4), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))
stale7=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 3), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))

stale_2=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('skip_connect', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 4), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))
zero_2=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))

throw60=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 4), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
use60=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('skip_connect', 2), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
stale60=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))


throw30=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('skip_connect', 1), ('avg_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2)], reduce_concat=range(2, 6))
use30=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))


stale3421=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 1), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))
throw3421=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('skip_connect', 2), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 3), ('avg_pool_3x3', 4), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))


throw3241=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 1), ('sep_conv_5x5', 3), ('max_pool_3x3', 4), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
use3241=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('max_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 3), ('dil_conv_5x5', 1)], reduce_concat=range(2, 6))

#dis
dishard=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))
dissoft=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 2), ('sep_conv_5x5', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))

#cleint
c2s5=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 3), ('avg_pool_3x3', 1), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))
c2s2=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
c2s7=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
c5s5=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 4), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 4), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))
c5s2=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 4), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 4), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))
c5s7=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('sep_conv_3x3', 4), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))

#noniid
non02359=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2), ('dil_conv_3x3', 2), ('max_pool_3x3', 1), ('sep_conv_5x5', 3), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
noniid=Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('dil_conv_5x5', 4), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 4), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))

#cifar100
cifar100=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('skip_connect', 2), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))

#cvpr 
cvprfed = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1),
            ('sep_conv_5x5', 3), ('dil_conv_5x5', 3), ('sep_conv_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0),
            ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
#svhn
svhn=Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('avg_pool_3x3', 2), ('dil_conv_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('avg_pool_3x3', 1), ('dil_conv_5x5', 3), ('dil_conv_3x3', 2), ('max_pool_3x3', 1), ('dil_conv_3x3', 0)], reduce_concat=range(2, 6))
svhn3=Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 3), ('sep_conv_5x5', 0), ('dil_conv_3x3', 4), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_5x5', 3), ('skip_connect', 1)], reduce_concat=range(2, 6))
svhn4=Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 4), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_5x5', 1), ('skip_connect', 2)], reduce_concat=range(2, 6))
svhn5=Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 2), ('dil_conv_3x3', 2), ('avg_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('skip_connect', 3), ('dil_conv_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))
svhns4=Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('skip_connect', 1), ('dil_conv_5x5', 3), ('skip_connect', 2), ('dil_conv_5x5', 0), ('max_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('skip_connect', 3), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 3)], reduce_concat=range(2, 6))

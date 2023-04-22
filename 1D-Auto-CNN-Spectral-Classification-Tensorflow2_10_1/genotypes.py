from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x1',
    'avg_pool_3x1',
    'skip_connect',
    'conv_3x1',
    'conv_5x1',
    'conv_7x1',
    'conv_9x1',
]

# 原版的HSI，Genotype是命名元组
# HSI = Genotype(normal=[('conv_7x1', 0), ('avg_pool_3x1', 1), ('conv_3x1', 0), ('conv_7x1', 1), ('conv_5x1', 1), ('conv_5x1', 3), ('conv_7x1', 3), ('conv_5x1', 2)], normal_concat=range(2, 6), reduce=[('conv_7x1', 0), ('avg_pool_3x1', 1), ('conv_7x1', 0), ('avg_pool_3x1', 2), ('conv_7x1', 0), ('avg_pool_3x1', 1), ('avg_pool_3x1', 1), ('skip_connect', 0)], reduce_concat=range(2, 6))
HSI = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('conv_7x1', 2), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('conv_9x1', 1), ('conv_5x1', 0), ('conv_7x1', 1), ('conv_3x1', 2), ('conv_7x1', 2), ('conv_7x1', 1), ('avg_pool_3x1', 2), ('conv_5x1', 1)], reduce_concat=range(2, 6))
# 2021-08-21 03:55:04 Searched Neural Architecture:
HSI = Genotype(normal=[('skip_connect', 0), ('conv_9x1', 1), ('conv_7x1', 2), ('conv_3x1', 1), ('conv_5x1', 0), ('conv_3x1', 3), ('conv_3x1', 0), ('conv_5x1', 2)], normal_concat=range(2, 6), reduce=[('conv_5x1', 0), ('avg_pool_3x1', 1), ('conv_3x1', 2), ('conv_3x1', 0), ('conv_7x1', 3), ('skip_connect', 2), ('conv_7x1', 4), ('conv_5x1', 0)], reduce_concat=range(2, 6))
# 2021-08-21 10:14:55 Searched Neural Architecture:
HSI = Genotype(normal=[('conv_5x1', 1), ('conv_3x1', 0), ('skip_connect', 2), ('conv_7x1', 0), ('conv_7x1', 0), ('conv_5x1', 3), ('conv_9x1', 3), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('conv_7x1', 0), ('conv_5x1', 1), ('conv_5x1', 2), ('conv_9x1', 1), ('avg_pool_3x1', 0), ('skip_connect', 1), ('avg_pool_3x1', 4), ('avg_pool_3x1', 2)], reduce_concat=range(2, 6))
# 2021-08-21 10:53:40 Searched Neural Architecture:
HSI = Genotype(normal=[('conv_9x1', 0), ('conv_7x1', 1), ('conv_7x1', 1), ('conv_7x1', 2), ('conv_3x1', 1), ('conv_5x1', 3), ('conv_7x1', 1), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('skip_connect', 1), ('conv_3x1', 0), ('conv_5x1', 2), ('skip_connect', 0), ('conv_7x1', 3), ('conv_3x1', 2), ('conv_5x1', 0)], reduce_concat=range(2, 6))

# 2021-08-21 11:24:42 Searched Neural Architecture:
HSI = Genotype(normal=[('conv_7x1', 1), ('conv_9x1', 0), ('conv_7x1', 1), ('skip_connect', 0), ('conv_3x1', 3), ('conv_9x1', 1), ('conv_3x1', 3), ('conv_9x1', 4)], normal_concat=range(2, 6), reduce=[('conv_7x1', 1), ('conv_5x1', 0), ('conv_9x1', 2), ('skip_connect', 1), ('conv_5x1', 1), ('conv_3x1', 2), ('conv_7x1', 1), ('avg_pool_3x1', 3)], reduce_concat=range(2, 6))

# 2021-08-26 00:39:51 Searched Neural Architecture:
HSI = Genotype(normal=[('max_pool_3x1', 0), ('conv_5x1', 1), ('max_pool_3x1', 0), ('conv_7x1', 2), ('conv_9x1', 3), ('conv_9x1', 1), ('skip_connect', 1), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x1', 1), ('conv_7x1', 0), ('skip_connect', 0), ('skip_connect', 2), ('max_pool_3x1', 1), ('max_pool_3x1', 0), ('max_pool_3x1', 1), ('max_pool_3x1', 3)], reduce_concat=range(2, 6))

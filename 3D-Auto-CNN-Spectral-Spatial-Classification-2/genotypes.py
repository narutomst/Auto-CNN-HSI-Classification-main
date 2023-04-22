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

HSI = Genotype(normal=[('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 2), ('max_pool_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 2), ('dil_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('skip_connect', 4)], reduce_concat=range(2, 6))
# 第一次训练10次的结果
# HSI = Genotype(normal=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))



# 2021-09-01 18:56:13 Searched Neural Architecture:
HSI = Genotype(normal=[('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

# 2021-09-11 17:01:06 Searched Neural Architecture:
HSI = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

from enum import Enum
from re import L


class MagneticField(Enum):
    BN_MINUS_10 = -1.0
    BN_MINUS_09 = -0.9
    BN_MINUS_08 = -0.8
    BN_MINUS_07 = -0.7000000000000001
    BN_MINUS_06 = -0.6000000000000001
    BN_MINUS_05 = -0.5000000000000001
    BN_MINUS_04 = -0.40000000000000013
    BN_MINUS_03 = -0.30000000000000016
    BN_MINUS_02 = -0.20000000000000018
    BN_MINUS_01 = -0.1000000000000002
    BN_0 = -2.220446049250313e-16
    BN_01 = 0.09999999999999964
    BN_02 = 0.19999999999999973
    BN_03 = 0.2999999999999998
    BN_04 = 0.3999999999999997
    BN_05 = 0.49999999999999956
    BN_06 = 0.5999999999999996
    BN_07 = 0.6999999999999997
    BN_08 = 0.7999999999999996
    BN_09 = 0.8999999999999995
    BN_10 = 0.09999999999999964


def create_bn(x, nqubit):
    return str([x.value] * nqubit)


def create_bn_list(nqubit):
    l = []
    for x in MagneticField:
        l.append(create_bn(x, nqubit))
    return l

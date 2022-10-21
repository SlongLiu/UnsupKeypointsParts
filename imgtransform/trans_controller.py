import torch
from .tps import randTPSTransform
from .spatialTransform import randSpatialTransform
from .tps_sampler import randTPStransform2

def get_trans_controller(TRANS_CONTROLLER_DICT):
    typename = TRANS_CONTROLLER_DICT.NAME
    assert typename in ['randTPSTransform', 'randSpatialTransform', 'randTPStransform2']
    tc = eval(typename)(**TRANS_CONTROLLER_DICT.PARAS)
    return tc



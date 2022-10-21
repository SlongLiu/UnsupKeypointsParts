from .perceptualloss import *
from .meshreconsloss import *
from .reconsloss import *
from .localfeatloss import *
from .concentration_loss import *
from .pixel_connection_loss import *

def get_metrics(metric_config_list, args=None):
    metrics = []
    for m in metric_config_list:
        for k in m.keys():
            assert k in ['name', 'paras', 'weight', 'alias']
        metric = eval(m['name'])(**m['paras'], args=args)
        metrics.append((
            m['name'],
            metric,
            m['weight'],
            m.get('alias', m['name'])
        ))
    # print('metrics:', metrics)
    # raise ValueError
    return metrics

import torch
import torch.nn as nn
try:
    from torch.cuda.amp import autocast
    autocast_label = True
except:
    print('No amp found! you torch version: %s, please update to 1.6.0 or newer!' % torch.__version__)
    from utils.utils import FeiWu
    autocast_label = False

from loss.get_metrics import get_metrics



from network.utils import initialize_weights


def get_basemodel(args):
    if args.MODEL.basename == 'MyIMM':
        from network.immmodel import MyIMM
        model = MyIMM(**args.MODEL.paras)
        initialize_weights(model)
    elif args.MODEL.basename == 'SCOPSP':
        from network.scopsplus import SCOPSP
        model = SCOPSP(**args.MODEL.paras)
    elif args.MODEL.basename == 'SCOPSP2':
        from network.scopsplus2 import SCOPSP2
        model = SCOPSP2(**args.MODEL.paras)
    elif args.MODEL.basename == 'MyIMMBN':
        from network.immmodel import MyIMMBN
        model = MyIMMBN(**args.MODEL.paras)
        initialize_weights(model)
    elif args.MODEL.basename == 'MyIMMPP':
        from network.immmodel import MyIMMPP
        model = MyIMMPP(**args.MODEL.paras)
        initialize_weights(model)
    elif args.MODEL.basename == 'IMM3D':
        from network.imm3d import IMM3D
        model = IMM3D(**args.MODEL.paras)
    elif args.MODEL.basename == 'IMM3DS':
        from network.imm3d import IMM3DS
        model = IMM3DS(**args.MODEL.paras)
    elif args.MODEL.basename == 'IMMSC':
        from network.immmodel import IMMSC
        model = IMMSC(**args.MODEL.paras)
    elif args.MODEL.basename == 'MeshNetPlus':
        from network.meshnet import MeshNetPlus
        model = MeshNetPlus(**args.MODEL.paras, args=args)
    else:
        raise NotImplementedError("%s is not implemented yet" % args.MODEL.NAME)

    return model


class ModelWarperV2(nn.Module):
    def __init__(self, args):
        super(ModelWarperV2, self).__init__()
        self.model = get_basemodel(args)
        self.set_metrics(args)
                
        self.mixed_precision = args.get('mixed_precision', False)

        # Compatible with previous versions
        if autocast_label:
            self.autocast_func = autocast
        else:
            self.autocast_func = FeiWu

    def set_metrics(self, args):
        self.metrics_train = get_metrics(args.metriclist_train, args=args)
        self.metrics_train_list = nn.ModuleList([x[1] for x in self.metrics_train])

        if args.get("metriclist_test", None) is None:
            self.no_metriclist_test = True
        else:
            self.no_metriclist_test = False
            self.metrics_test = get_metrics(args.metriclist_test, args=args)
            self.metrics_test_list = nn.ModuleList([x[1] for x in self.metrics_train])

    def get_model_output(self, ddata, mode):
        return self.model(ddata, mode)


    def get_loss(self, ddata):
        if (self.training) or ((not self.training) and (self.no_metriclist_test)):
            metrics = self.metrics_train
            metrics_list = self.metrics_train_list
        else:
            metrics = self.metrics_test
            metrics_list = self.metrics_test_list

        lossdict = {}
        loss = 0
        for i in range(len(metrics)):
            m_name, _, weight, alias = metrics[i]
            metric = metrics_list[i]

            # metric
            loss_i = metric(ddata)
            if not isinstance(loss_i, dict):
                loss_i = {alias: loss_i}

            # add sum
            for k,v in loss_i.items():
                loss = loss + weight * v
            if len(loss_i) == 1:
                lossdict.update({k: v * weight for k,v in loss_i.items()})
            else:
                lossdict.update({(alias + '-' + k): v * weight for k,v in loss_i.items()})

            # debug
            # print(m_name, loss_i)

        res = {
            'loss': loss,
            'lossdict': lossdict
        }

        # print('loss:', loss)
        # print('lossdict:', lossdict)

        return res


    def forward(self, ddata, mode='full_pipeline', get_loss=True):
        with self.autocast_func(enabled=self.mixed_precision):
            model_output = self.get_model_output(ddata, mode)
            ddata.update({'output': model_output})
            if get_loss:
                loss_output = self.get_loss(ddata)
                ddata.update({'loss': loss_output})

        return ddata  
        
    def load(self, path, map_location='cpu', strict=True):
        print("Loading checkpoint from '{}'".format(path), flush=True)
        checkpoint = torch.load(path, map_location = map_location) # lambda storage, loc: storage.cuda(gpuid)
        # load state dict to base model
        self.model.load_state_dict(checkpoint['state_dict'], strict=strict)

    def get_paras(self, methodname):
        from itertools import chain 
        paras1 = self.metrics_train_list.parameters()
        paras2 = getattr(self.model, methodname)()
        return chain(paras1, paras2)

##################################depreciated####################

class ModelWarper(nn.Module):
    def __init__(self, args):
        super(ModelWarper, self).__init__()
        self.model = get_basemodel(args)
        self.metrics_train = get_metrics(args.metriclist_train, args=args)
        for item in self.metrics_train:
            super(ModelWarper, self).add_module('metric_train_' + item[0], item[1])

        if args.get("metriclist_test", None) is None:
            self.metrics_test = self.metrics_train
        else:
            self.metrics_test = get_metrics(args.metriclist_test, args=args)
            for item in self.metrics_test:
                super(ModelWarper, self).add_module('metric_test_' + item[0], item[1])
                
        self.mixed_precision = args.get('mixed_precision', False)

    def get_model_output(self, ddata, mode):
        return self.model(ddata, mode)

    def get_loss(self, ddata):
        metrics = self.metrics_train if self.training else self.metrics_test
        lossdict = {}
        loss = 0
        for m_name, metric, weight, alias in metrics:
            loss_i = metric(ddata)
            if not isinstance(loss_i, dict):
                loss_i = {alias: loss_i}
            # print('loss_i:', loss_i)
            # add sum
            for k,v in loss_i.items():
                loss = loss + weight * v
            if len(loss_i) == 1:
                lossdict.update({k: v * weight for k,v in loss_i.items()})
            else:
                lossdict.update({(alias + '-' + k): v * weight for k,v in loss_i.items()})

        res = {
            'loss': loss,
            'lossdict': lossdict
        }

        return res

    def forward(self, ddata, mode='full_pipeline', get_loss=True):
        with autocast(enabled=self.mixed_precision):
            model_output = self.get_model_output(ddata, mode)
            ddata.update({'output': model_output})
            if get_loss:
                loss_output = self.get_loss(ddata)
                ddata.update({'loss': loss_output})

        return ddata

    def load(self, path, map_location='cpu', strict=True):
        print("Loading checkpoint from '{}'".format(path), flush=True)
        checkpoint = torch.load(path, map_location = map_location) # lambda storage, loc: storage.cuda(gpuid)
        # load state dict to base model
        self.model.load_state_dict(checkpoint['state_dict'], strict=strict)

    def get_paras(self, methodname):
        from itertools import chain 
        paras1 = self.metrics_train_list.parameters()
        paras2 = getattr(self.model, methodname)()
        return chain(paras1, paras2)
        
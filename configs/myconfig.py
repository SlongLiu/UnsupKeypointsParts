import argparse
import torch
import os, sys
import os.path as osp

from utils.slconfig import SLConfig, DictAction


def parse_args():
    """
    Reads the command switches and creates a config
    Command line switches override config files

    :return: args
    """

    """ config """
    parser = argparse.ArgumentParser(description='Parameters can be set by -c config.py or by positional params.')

    parser.add_argument('--device', '-d', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--save_dir', '-s', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--gpu_devices', '-g', type=str)

    """ model parameters """
    parser.add_argument('--heatmap_std', type=int)
    parser.add_argument('--num_keypoints', type=int)
    parser.add_argument('--pretrained', type=str)

    """ hyper-parameters """
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--batch_size', '-b', type=int)
    parser.add_argument('--lr', '-l', type=float)

    """ data and data augmentation parameters """
    parser.add_argument('--dataset', type=str)

    parser.add_argument('--data_rescale_height', type=float)
    parser.add_argument('--data_rescale_width', type=float)
    parser.add_argument('--trainratio', type=float)

    """ set config files """
    parser.add_argument('--config', '-c', help='train config file path')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument('--testmode', action='store_true', help='Test program integrity')
    parser.add_argument('--resume', type=str, help='resumed model path')
    parser.add_argument('--first-test', type=bool, default=False, help='first-test')

    args = parser.parse_args()
    
    # set gpu env
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices

    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    
    # store the command used
    args.command = ' '.join(sys.argv)
    return args

def get_config(copycfg=True):
    args = parse_args()

    cfg = SLConfig.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    v_args = {}
    for k,v in vars(args).items():
        if k == 'options' or v is None:
            continue
        v_args[k] = v
    cfg.merge_from_dict(v_args)

    # for test
    if cfg.get('just_test', False):
        _config_dict_dummy = {
            'skip_train': 1,
            'skip_test': 1,
            'skip_plot': 1,
            'TRAIN.epochs': 1
        }
        cfg.merge_from_dict(_config_dict_dummy)

    # copy config file
    if copycfg:
        if 'save_dir' in cfg:
            cfgsave = cfg.copy()
            this_dir = osp.dirname(__file__)
            cfgsave.device = 'cuda'
            traget_file_name = osp.join(osp.dirname(this_dir), cfgsave.save_dir, 'config-' + osp.basename(cfgsave.config))
            os.makedirs(osp.dirname(traget_file_name), exist_ok=True)
            cfgsave.dump(traget_file_name)

    return cfg

if __name__ == '__main__':
    t = get_config()
    print(t)
    print(vars(t))
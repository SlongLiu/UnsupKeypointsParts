save_dir = None

img_size = (128, 128)
batch_size = 1024

seed = 123
nodes = 1

lm_numb =  10 # number of landmarks to detection
mixed_precision = True

MASTER_ADDR = '127.0.0.1'
MASTER_PORT = '9786'


normalize_mean = [0.0] * 3
normalize_std = [1.0] * 3
data_transform_pipe = [
            dict(
                name = 'SLResize',
                paras = dict(size=(int(img_size[0]/0.8),int(img_size[1]/0.8)))
            ),
            dict(
                name = 'SLCenterCrop',
                paras = dict(size=img_size)
            ),
            dict(name='SLToTensor', paras={}),
            dict(
                name = 'SLNormalize',
                paras = dict(mean=normalize_mean, std=normalize_std)
            ),
        ]

# dataset
DATASET = dict(
    TRAIN_SET = dict(
        # for dataset
        dataset = 'celeba',
        paras = dict(
            root_dir = '/data/shilong/data/imm/celeba/Img/img_align_celeba_hq' ,
            select_path = None, # training
            anno_path = None,
            pic_trans_num = 0,
            json_file = '/data/shilong/data/imm/celeba/2d_lm_pred/lm_pred_ep0.json'
        ),

        data_rescale_height = img_size[0],
        data_rescale_width = img_size[1],
        normalize_mean = normalize_mean,
        normalize_std = normalize_std,

        # for loader
        batch_size = batch_size,
        num_workers = 20,

        # for pipeline
        Tname = 'SLT',
        pipeline = data_transform_pipe
    ),
)

# model
MODEL = dict(
    NAME = 'IMM3D',
    basename = 'IMM3D',
    paras = dict(
        nz_feat=128, 
        n_kp=10, 
        input_shape=(128, 128), 
        mean_v_path=None, 
        mean_v_fix=False
    )
)

# loss
metriclist_train = [
    {
        'name': 'L2NormLoss',
        'paras': {
            'item_map': dict(x='output.delta_v'),
        },
        'weight': 1.0
    },
    {
        'name': 'keyPointsLoss',
        'paras': {
            'item_map': dict(x='lm', y='output.pred_v_2d'),
        },
        'weight': 1e-3
    },    
]

metriclist_test = None # the same as metriclist_train

# train
TRAIN = dict(
    epochs = 100,
    ntrain_per_epoch = 5,

    batch_size = batch_size,
    optimizer = 'Adam',
    lr = 1e-4,
    plot_interval = 1,

    print_interval_train = 50,
    weight_decay = 5e-4,
)

M_step_conf = dict(
    # save_folder='/data/shilong/data/imm/celeba/2d_lm_pred',
    # save_folder = ''
    lr = 1e-3
)
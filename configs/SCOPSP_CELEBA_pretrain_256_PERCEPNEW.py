save_dir = None

heatmap_size = (32, 32)
img_size = (256, 256)
batch_size = 24

seed = 123
nodes = 1

lm_numb = 10 # number of landmarks to detection
mixed_precision = True

MASTER_ADDR = '127.0.0.1'
MASTER_PORT = '9786'

# transformer
TRANS_CONTROLLER = dict(
    NAME = 'randTPStransform2',
    PARAS = dict(
        height = 256,
        width = 256,
    )
)

normalize_mean = [0.0] * 3
normalize_std = [1.0] * 3
data_transform_pipe = [
            dict(
                name = 'SLRandomRotation',
                paras = dict(degrees=(-0.5, 0.5))
            ),
            dict(
                name = 'SLRandomResizedCrop',
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
            pic_trans_num = 0, pic_trans_type = TRANS_CONTROLLER, pic_return_mask = True, pic_trans_cont=False, soft_mask=False,
        ),

        data_rescale_height = img_size[0],
        data_rescale_width = img_size[1],
        normalize_mean = normalize_mean,
        normalize_std = normalize_std,

        # for loader
        batch_size = batch_size,
        num_workers = 16,

        # for pipeline
        Tname = 'SLT',
        pipeline = data_transform_pipe
    ),
)

# model
MODEL = dict(
    NAME = 'SCOPSP',
    basename = 'SCOPSP',
    paras = dict(
        dim = lm_numb,
        in_channel = 3,
        h_channel = 32,
        mode = 'point',
        heatmap_std = 0.02,
        downsamplelast = True,
    )
)

# loss
metriclist_train = [
    {
        'name': 'PerceptualNewLoss',
        'paras': {
            'model_type': 'Vgg19',
            'content_layer': [0, 1, 2, 3, 4, 5],
            'loss_type': 'L2',
            'layer_weight': [1/0.25, 1/0.26, 1/1.4, 1/3.19, 1/2.54, 1/0.3],
            'item_map': [
                dict(x='img', y='output.img_rec'),
            ],
            'refresh_iterval': 100,
        },
        'weight': 1.0
    },
]

metriclist_test = None # the same as metriclist_train

SCHEDULER = dict(
    name = 'CosineAnnealingLR',
    paras = dict(
        T_max = 100
    )
)

# train
TRAIN = dict(
    epochs = 100,
    batch_size = batch_size,
    optimizer = 'Adam',
    lr = 1e-4,
    plot_interval = 1,
    test_interval = 1,

    print_interval_train = 50,
    weight_decay = 5e-4,

    only_save_best_model = False,
    tb_plot_interval = 1000
)


